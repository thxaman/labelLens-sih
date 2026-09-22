import io
import time
import base64
import logging
from typing import Union, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR

from schemas.ocr import (
    OCRScanResult,
    TextBlock,
    ImageMetadata,
    BBox,
    BlockSize,
    Point
)

logger = logging.getLogger("ocr_service")


def calculate_bbox_iou(bbox1: BBox, bbox2: BBox) -> float:
    """Computes Intersection over Union between two axis-aligned bounding boxes."""
    x_left = max(bbox1.x_min, bbox2.x_min)
    y_top = max(bbox1.y_min, bbox2.y_min)
    x_right = min(bbox1.x_max, bbox2.x_max)
    y_bottom = min(bbox1.y_max, bbox2.y_max)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bbox1.x_max - bbox1.x_min) * (bbox1.y_max - bbox1.y_min)
    area2 = (bbox2.x_max - bbox2.x_min) * (bbox2.y_max - bbox2.y_min)
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0
    return float(intersection_area / union_area)


class OCRService:
    def __init__(self):
        """
        Initialize RapidOCR engine running on ONNXRuntime CPU.
        Runs smoothly on standard laptops without GPU requirements.
        """
        try:
            self.engine = RapidOCR()
            logger.info("RapidOCR engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RapidOCR engine: {e}")
            raise RuntimeError(f"OCR Engine initialization error: {e}")

    def preprocess_standard(self, img_np: np.ndarray, enhance: bool = True) -> tuple[np.ndarray, float]:
        """
        Pipeline 1 (Standard): Auto-upscaling (<800px) + CLAHE + mild sharpening kernel.
        """
        height, width = img_np.shape[0], img_np.shape[1]
        scale_factor = 1.0

        # Cap oversized photos to max dimension 2048px for CPU inference speed (scales back automatically)
        max_dim = max(width, height)
        if max_dim > 2048:
            down_scale = 2048.0 / max_dim
            scale_factor = down_scale
            new_w = int(width * down_scale)
            new_h = int(height * down_scale)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif width < 800 or height < 800:
            scale_factor = 2.5 if width < 400 else 2.0
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        if enhance:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
            img_np = cv2.filter2D(img_np, -1, kernel)

        return img_np, scale_factor

    def preprocess_aggressive_contrast(self, img_np: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Pipeline 2 (Aggressive Contrast): 3x upscale + strong CLAHE (clipLimit=4.0) + unsharp mask.
        Ideal for low-contrast, faded, or reflective packaging labels.
        """
        height, width = img_np.shape[0], img_np.shape[1]
        max_dim = max(width, height)
        if max_dim > 2048:
            scale_factor = 2048.0 / max_dim
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scale_factor = 2.0 if width < 800 else (1.5 if width < 1200 else 1.0)
            if scale_factor > 1.0:
                new_w = int(width * scale_factor)
                new_h = int(height * scale_factor)
                img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        # Strong CLAHE on Luminance channel
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # Unsharp masking (original * 1.5 - blur * 0.5)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        unsharp = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        return unsharp, scale_factor

    def preprocess_otsu_binarize(self, img_np: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Pipeline 3 (Otsu Binarization): Denoising + Otsu adaptive thresholding.
        Isolates dark printed characters on light backgrounds cleanly.
        """
        height, width = img_np.shape[0], img_np.shape[1]
        scale_factor = 2.0 if width < 1000 else 1.0

        if scale_factor > 1.0:
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.copy()

        # Denoise before thresholding
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rgb_bin = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        return rgb_bin, scale_factor

    def preprocess_inverted(self, img_np: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Pipeline 4 (Inverted Grayscale): Inverts colors for light-on-dark labels (e.g. white text on black/blue).
        RapidOCR is trained primarily on dark-on-light text; inverting dramatically increases recall.
        """
        height, width = img_np.shape[0], img_np.shape[1]
        scale_factor = 2.0 if width < 1000 else 1.0

        if scale_factor > 1.0:
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.copy()

        inverted = cv2.bitwise_not(gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        inverted_clahe = clahe.apply(inverted)
        rgb_inv = cv2.cvtColor(inverted_clahe, cv2.COLOR_GRAY2RGB)
        return rgb_inv, scale_factor

    def preprocess_image(self, img_np: np.ndarray, enhance: bool = True) -> tuple[np.ndarray, float]:
        """Backwards-compatible alias for standard preprocessing."""
        return self.preprocess_standard(img_np, enhance=enhance)

    def load_image(self, image_input: Union[str, bytes, Image.Image, np.ndarray]) -> tuple[np.ndarray, ImageMetadata]:
        """
        Normalizes any supported image input into numpy RGB array and ImageMetadata.
        """
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
            img_np = np.array(pil_img)
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            img_np = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            img_np = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        height, width = img_np.shape[0], img_np.shape[1]
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        metadata = ImageMetadata(width=width, height=height, channels=channels)

        return img_np, metadata

    def _run_engine_on_image(
        self,
        processed_img: np.ndarray,
        scale_factor: float,
        min_confidence: float
    ) -> List[TextBlock]:
        """Runs RapidOCR on a preprocessed image and transforms coordinates back to original scale."""
        ocr_result, _ = self.engine(processed_img)
        blocks: List[TextBlock] = []
        if not ocr_result:
            return blocks

        for idx, item in enumerate(ocr_result):
            box, text, score = item[0], item[1], float(item[2])

            if score < min_confidence or not text.strip():
                continue

            polygon = [[float(pt[0]) / scale_factor, float(pt[1]) / scale_factor] for pt in box]
            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]

            x_min, x_max = float(min(xs)), float(max(xs))
            y_min, y_max = float(min(ys)), float(max(ys))

            width = max(x_max - x_min, 1.0)
            height = max(y_max - y_min, 1.0)
            aspect_ratio = round(width / height, 2)
            font_size_px = round(height, 1)

            center_x = round((x_min + x_max) / 2.0, 1)
            center_y = round((y_min + y_max) / 2.0, 1)

            block = TextBlock(
                id=idx + 1,
                text=text.strip(),
                confidence=round(score, 4),
                polygon=polygon,
                bbox=BBox(
                    x_min=round(x_min, 1),
                    y_min=round(y_min, 1),
                    x_max=round(x_max, 1),
                    y_max=round(y_max, 1)
                ),
                size=BlockSize(
                    width=round(width, 1),
                    height=round(height, 1),
                    aspect_ratio=aspect_ratio,
                    estimated_font_size_px=font_size_px
                ),
                center=Point(x=center_x, y=center_y)
            )
            blocks.append(block)

        return blocks

    def _merge_blocks(
        self,
        base_blocks: List[TextBlock],
        new_blocks: List[TextBlock],
        iou_threshold: float = 0.45
    ) -> List[TextBlock]:
        """
        Deduplicates newly discovered text blocks against existing ones using IoU.
        Upgrades lower-confidence blocks when higher-confidence or longer text is found.
        """
        merged = list(base_blocks)

        for new_b in new_blocks:
            is_dup = False
            for i, existing in enumerate(merged):
                iou = calculate_bbox_iou(new_b.bbox, existing.bbox)
                if iou > iou_threshold:
                    is_dup = True
                    # Upgrade if new block has better confidence and equivalent or longer text
                    if new_b.confidence > existing.confidence and len(new_b.text) >= len(existing.text):
                        merged[i] = new_b
                    break

            if not is_dup:
                merged.append(new_b)

        # Sort merged blocks in reading order (top-to-bottom, left-to-right)
        merged.sort(key=lambda b: (round(b.bbox.y_min / 20.0), b.bbox.x_min))
        for i, b in enumerate(merged):
            b.id = i + 1

        return merged

    def extract_text(
        self,
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        enhance: bool = False,
        min_confidence: float = 0.3,
        include_annotated_image: bool = False,
        fallback: bool = True
    ) -> OCRScanResult:
        """
        Multi-pipeline OCR extraction function with Early-Exit and IoU Deduplication.

        Pipelines:
          1. Standard (auto-upscale + CLAHE + mild sharpen)
          2. Aggressive Contrast (3x upscale + strong CLAHE + unsharp mask)
          3. Otsu Binarization (adaptive thresholding)
          4. Inverted Grayscale (white text on dark backgrounds)

        Early-exit condition:
          If Pipeline 1 finds >= 5 text blocks with mean confidence >= 0.70,
          it returns immediately without running fallback pipelines.
        """
        start_time = time.time()
        try:
            img_np, metadata = self.load_image(image_input)

            # Pipeline 1: Standard
            p1_img, s1 = self.preprocess_standard(img_np, enhance=enhance)
            blocks = self._run_engine_on_image(p1_img, s1, min_confidence)
            pipelines_run = ["standard"]

            # Evaluate Early-Exit Condition:
            mean_conf = (sum(b.confidence for b in blocks) / len(blocks)) if blocks else 0.0
            # If standard pass already found good text, don't trigger 3 expensive fallback OCR passes on CPU
            has_sufficient_text = (len(blocks) >= 3 and mean_conf >= 0.60) or (len(blocks) >= 1 and mean_conf >= 0.75)
            should_fallback = fallback and not has_sufficient_text and (len(blocks) < 2 or mean_conf < 0.50)

            if should_fallback:
                logger.info(
                    "OCR fallback triggered (initial blocks=%d, mean_conf=%.2f)",
                    len(blocks),
                    mean_conf
                )
                fallback_stages = [
                    ("aggressive_contrast", self.preprocess_aggressive_contrast),
                    ("otsu_binarize", self.preprocess_otsu_binarize),
                    ("inverted", self.preprocess_inverted),
                ]

                for name, prep_func in fallback_stages:
                    # Enforce a 12-second total time budget so OCR never causes a gateway timeout
                    if (time.time() - start_time) > 12.0:
                        logger.warning("OCR fallback reached time budget (%.2fs), returning detected blocks", time.time() - start_time)
                        break

                    pipelines_run.append(name)
                    fb_img, fb_scale = prep_func(img_np)
                    fb_blocks = self._run_engine_on_image(fb_img, fb_scale, min_confidence)
                    blocks = self._merge_blocks(blocks, fb_blocks)

                    # Check if candidate pool reached healthy threshold
                    curr_conf = (sum(b.confidence for b in blocks) / len(blocks)) if blocks else 0.0
                    if len(blocks) >= 4 and curr_conf >= 0.65:
                        break

            raw_lines = [b.text for b in blocks]
            raw_text = "\n".join(raw_lines)
            processing_time = round((time.time() - start_time) * 1000, 2)

            annotated_b64 = None
            if include_annotated_image:
                annotated_b64 = self.generate_annotated_image(img_np, blocks)

            return OCRScanResult(
                success=True,
                image_metadata=metadata,
                total_text_blocks=len(blocks),
                text_blocks=blocks,
                raw_text=raw_text,
                processing_time_ms=processing_time,
                annotated_image_base64=annotated_b64,
                pipelines_executed=pipelines_run
            )

        except Exception as e:
            logger.error(f"Error during OCR extraction: {e}", exc_info=True)
            processing_time = round((time.time() - start_time) * 1000, 2)
            return OCRScanResult(
                success=False,
                image_metadata=ImageMetadata(width=0, height=0, channels=0),
                total_text_blocks=0,
                text_blocks=[],
                raw_text="",
                processing_time_ms=processing_time,
                error=str(e),
                pipelines_executed=[]
            )

    def generate_annotated_image(self, img_np: np.ndarray, text_blocks: List[TextBlock]) -> str:
        """
        Draws bounding boxes, text labels, and font sizes on the image for visual verification.
        Returns base64 encoded JPEG image string.
        """
        pil_img = Image.fromarray(img_np).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        for block in text_blocks:
            pts = [(pt[0], pt[1]) for pt in block.polygon]
            draw.polygon(pts, outline=(0, 102, 204, 255), fill=(0, 102, 204, 40))

            x_min, y_min = block.bbox.x_min, block.bbox.y_min
            label = f"#{block.id} {block.text} ({block.size.estimated_font_size_px}px)"
            draw.text((x_min, max(0, y_min - 14)), label, fill=(255, 0, 0, 255))

        combined = Image.alpha_composite(pil_img, overlay).convert("RGB")
        buffered = io.BytesIO()
        combined.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Module-level singleton instance
_ocr_service_instance: Optional[OCRService] = None

def get_ocr_service() -> OCRService:
    global _ocr_service_instance
    if _ocr_service_instance is None:
        _ocr_service_instance = OCRService()
    return _ocr_service_instance

def extract_text_from_image(
    image_input: Union[str, bytes, Image.Image, np.ndarray],
    enhance: bool = False,
    min_confidence: float = 0.3,
    include_annotated_image: bool = False,
    fallback: bool = True
) -> OCRScanResult:
    service = get_ocr_service()
    return service.extract_text(
        image_input,
        enhance=enhance,
        min_confidence=min_confidence,
        include_annotated_image=include_annotated_image,
        fallback=fallback
    )
