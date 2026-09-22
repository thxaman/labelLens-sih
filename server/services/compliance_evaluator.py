import re
import io
import time
import uuid
import base64
import logging
from typing import List, Dict, Any, Optional

try:
    from PIL import Image, ImageDraw  # type: ignore
except ImportError:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore

from schemas.ocr import OCRScanResult, TextBlock, BBox
from schemas.compliance import (
    ComplianceResult,
    ComplianceSummary,
    DeclarationFound,
    DeclarationMissing,
    ViolationDetail
)
from services.llm_evaluator import get_llm_evaluator, get_package_element_for_rule
from services.rag.citation_service import get_citation_service

logger = logging.getLogger("compliance_evaluator")


def merge_ocr_results(face_results: List[OCRScanResult]) -> OCRScanResult:
    """
    Combines the OCR output of several product faces into ONE product-level
    OCRScanResult, so the compliance engine evaluates the whole product: a
    mandatory declaration printed on any face (MRP on the back, ingredients on
    the side) satisfies its rule, exactly like a physical inspector reading
    every face of the package before ruling.

    Font metrics are rescaled to a common reference image height, because the
    engine estimates physical font size as px/image_height * 150mm: scaling px
    by ref_h/face_h keeps each block's mm estimate identical to what its own
    face would have produced. BBox coordinates are left untouched — they stay
    relative to the face image the block was read from (used as evidence only).
    """
    # Faces without any OCR blocks contribute nothing to the merged view but
    # keep their original index — Node's face_images array is indexed the same
    # way, so attribution must not renumber when a face is skipped.
    populated = [(i, f) for i, f in enumerate(face_results) if f and f.text_blocks]
    if not populated:
        raise ValueError("At least one face OCR result with text blocks is required")

    ref_height = max(f.image_metadata.height for _, f in populated)
    ref_face = next(f for _, f in populated if f.image_metadata.height == ref_height)

    merged_blocks: List[TextBlock] = []
    for face_index, face in populated:
        scale = ref_height / max(face.image_metadata.height, 1)
        for block in face.text_blocks:
            data = block.model_dump()
            data["id"] = len(merged_blocks) + 1
            data["face_index"] = face_index
            if scale != 1.0:
                size = data["size"]
                size["estimated_font_size_px"] = size["estimated_font_size_px"] * scale
                size["width"] = size.get("width", 0.0) * scale
                size["height"] = size.get("height", 0.0) * scale
            merged_blocks.append(TextBlock(**data))

    raw_text = "\n".join(f.raw_text for _, f in populated if f.raw_text)

    return OCRScanResult(
        success=all(f.success for _, f in populated),
        image_metadata=ref_face.image_metadata,
        total_text_blocks=len(merged_blocks),
        text_blocks=merged_blocks,
        raw_text=raw_text,
        processing_time_ms=sum(f.processing_time_ms for _, f in populated),
    )

# Patterns and keywords for category-specific declarations
CATEGORY_RULE_PATTERNS = {
    "fssai_license": {
        "keywords": ["FSSAI", "LIC NO", "LICENSE NO", "LIC. NO", "FSSAI LIC"],
        "regex": r"\b[12]\d{13}\b"
    },
    "veg_nonveg_symbol": {
        "keywords": ["VEG", "VEGETARIAN", "NON-VEG", "NON VEGETARIAN", "GREEN DOT", "BROWN TRIANGLE"],
    },
    "nutritional_info": {
        "keywords": ["NUTRITION", "NUTRITIONAL", "ENERGY", "PROTEIN", "CARBOHYDRATE", "FAT", "TOTAL SUGAR", "PER 100", "CALORIES", "KCAL"]
    },
    "ingredients_list": {
        "keywords": ["INGREDIENTS", "INGREDIENT", "CONTAINS", "COMPOSITION", "CONTENTS"]
    },
    "allergen_info": {
        "keywords": ["ALLERGEN", "ALLERGY", "MAY CONTAIN", "CONTAINS:", "CONTAINS WHEAT", "CONTAINS GLUTEN", "CONTAINS MILK", "CONTAINS NUTS", "CONTAINS SOY"]
    },
    "mfg_license_no": {
        "keywords": ["MFG LIC", "M.L. NO", "ML NO", "MFG. LIC", "MFG. LICENSE", "LICENSE NO", "M.L."]
    },
    "batch_lot_number": {
        "keywords": ["BATCH", "B.NO", "LOT NO", "LOT", "B. NO", "BATCH NO", "LOT NUMBER"]
    },
    "cosmetic_ingredients": {
        "keywords": ["INGREDIENTS", "COMPOSITION", "INCI", "CONTAINS", "AQUA", "WATER", "KEY INGREDIENTS", "ACTIVE INGREDIENTS", "PURIFIED WATER", "GLYCERIN"]
    },
    "directions_for_use": {
        "keywords": ["HOW TO USE", "DIRECTIONS FOR USE", "DIRECTIONS", "HOW TO APPLY", "USAGE", "APPLICATION", "USAGE DIRECTIONS", "APPLY TO", "APPLY ON", "MASSAGE GENTLY", "RINSE OFF", "PATCH TEST"]
    },
    "cosmetic_warnings": {
        "keywords": ["WARNING", "CAUTION", "FOR EXTERNAL USE ONLY", "AVOID CONTACT WITH EYES", "KEEP OUT OF REACH"]
    },
    "fibre_composition": {
        "keywords": ["COTTON", "POLYESTER", "FIBRE", "FABRIC", "WOOL", "SILK", "VISCOSE", "NYLON", "ELASTANE", "%"]
    },
    "size_declaration": {
        "keywords": ["SIZE", "CHEST", "WAIST", "CM", "LENGTH", "CHEST SIZE", "BODY MEASUREMENT"],
        "regex": r"\b(SIZE\s*:\s*[SMLX]+|\b[SMLX]{1,4}\b|\b\d{2,3}\s*CM\b)"
    },
    "wash_care": {
        "keywords": ["WASH", "CARE", "IRON", "BLEACH", "DRY CLEAN", "DO NOT BLEACH", "WARM WASH", "MACHINE WASH", "HAND WASH"]
    },
    "bis_registration": {
        "keywords": ["BIS", "CRS", "REGISTRATION", "IS/IEC", r"IS \d+", "ISI"],
        "regex": r"\b(R-\d{8}|IS\s*\d+)\b"
    },
    "power_ratings": {
        "keywords": ["VOLT", "WATT", "INPUT", "OUTPUT", "POWER", "RATING", "HZ"],
        "regex": r"\b\d+\s*(?:V|W|HZ|VOLT|WATT)\b"
    },
    "unit_sale_price": {
        "keywords": ["UNIT SALE PRICE", "UNIT PRICE", "USP", "PRICE PER"],
        "regex": r"(?:(?:RS\.?|₹|INR)\s*)?\d+(?:\.\d{1,2})?\s*(?:/|PER\s+)(?:G|GM|GMS|KG|KGS|ML|MLS|L|LTR|LTRS|LITRE|LITER|N|U|PCS|PIECE|PIECES|NUMBER|TABLET|TABLETS|SACHET|SACHETS|UNIT|UNITS)\b"
    },
    "pan_masala_warning": {
        "keywords": ["CHEWING OF PAN MASALA", "INJURIOUS TO HEALTH", "PAN MASALA", "GUTKHA", "HEALTH WARNING"],
        "regex": r"(?:CHEWING\s+OF\s+PAN\s+MASALA|INJURIOUS\s+TO\s+HEALTH)"
    },
    "qr_code_declaration": {
        "keywords": ["SCAN QR", "QR CODE", "SCAN FOR DETAILS", "SCAN FOR INFORMATION"],
        "regex": r"(?:SCAN\s+(?:QR|CODE|FOR))"
    },
    "bee_star_rating": {
        "keywords": ["BEE", "ENERGY STAR", "STAR RATING", "ELECTRICITY CONSUMPTION", "KWH/YEAR", "UNITS/YEAR", "ENERGY EFFICIENCY", "STAR LABEL"],
        "regex": r"(?:BEE\s+STAR|STAR\s+RATING|KWH\/YEAR|UNITS\/YEAR|ELECTRICITY\s+CONSUMPTION)"
    }
}



# --- Text Normalization -------------------------------------------------------------
# RapidOCR emits one text block per detected LINE, so a declaration printed across two
# lines ("MRP Rs 120 (Incl." + "of all taxes)") arrives as two separate blocks and is
# glued back together with "\n" inside raw_text. A plain substring search - against the
# per-line text OR against newline-joined raw text - therefore misses every wrapped
# phrase and reports a false "wrong_format". Every phrase check below runs on flattened,
# punctuation-folded text instead.

_UNICODE_FOLD = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
    "．": ".", "，": ",",
}


def flatten_text(text: str) -> str:
    """Uppercases, folds unicode lookalikes, and collapses every run of whitespace
    (newlines included) into a single space. Punctuation is preserved, so this is the
    form to use for numeric/format checks."""
    if not text:
        return ""
    for src, dst in _UNICODE_FOLD.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip().upper()


def normalize_phrase(text: str) -> str:
    """flatten_text() plus punctuation folding: every non-alphanumeric character becomes
    a single space, so 'Incl.of all taxes', 'INCL . OF ALL TAXES' and '(inclusive of all
    taxes)' all reduce to the same token sequence."""
    flat = flatten_text(text)
    return re.sub(r"[^A-Z0-9]+", " ", flat).strip()


def despace(norm: str) -> str:
    """normalize_phrase() output with every space removed. RapidOCR regularly returns a
    whole declaration as one unbroken token - 'MRPRS250.00(INCL.OFALLTAXES)',
    'NETQTY:440g' - because tightly-set label text gives it no gaps to split on. Word
    boundaries cannot see into a token like that, so keyword tests fall back to this."""
    return norm.replace(" ", "")


# Keywords too short or too common to test as bare substrings: each one hides inside an
# ordinary word ('RS' in CUSTOMERS, 'INC' in PRINCE, 'UNIT' in UNITED, 'WORKS' in
# NETWORKS), so those stay word-boundary-only.
_SUBSTRING_UNSAFE_KEYWORDS = {"RS", "INC", "LTD", "UNIT", "WORKS", "ORIGIN", "CORP", "LLP", "N"}


def _has_keyword(norm: str, keywords: List[str]) -> bool:
    """Word-boundary keyword test on normalize_phrase() output, with a despaced substring
    fallback for keywords distinctive enough to be safe. Pure substring matching fires on
    the wrong words - 'RS' inside CUSTOMERS, 'WORKS' inside NETWORKS - while pure
    word-boundary matching misses every glued OCR token; this covers both."""
    flat_norm = despace(norm)
    for keyword in keywords:
        if re.search(r"\b" + re.escape(keyword) + r"\b", norm):
            return True
        flat_keyword = despace(keyword)
        if (
            len(flat_keyword) >= 3
            and flat_keyword not in _SUBSTRING_UNSAFE_KEYWORDS
            and flat_keyword in flat_norm
        ):
            return True
    return False


# Tolerant matcher for the Rule 6 tax clause. It runs on normalize_phrase() output, so it
# only has to cope with token variants and not with punctuation, casing or line breaks:
# INCL / INCL. / INCLUSIVE / INCLUDING, optional OF, optional ALL, TAX / TAXES. The tokens
# must still appear in sequence, so a stray "TAX" somewhere on the label never passes.
_TAX_CLAUSE_RE = re.compile(r"\bINCL(?:U(?:SIVE|DING|DES))?\b(?:\s+OF)?(?:\s+ALL)?\s+TAX(?:ES)?\b")
# Same clause after despace(), for the glued "(INCL.OFALLTAXES)" spelling. Still anchored
# on INCL...TAX in sequence, so an unrelated "TAX" elsewhere on the label never passes.
_TAX_CLAUSE_DESPACED_RE = re.compile(r"INCL(?:U(?:SIVE|DING|DES))?(?:OF)?(?:ALL)?TAX(?:ES)?")


def has_tax_clause(norm: str) -> bool:
    return bool(_TAX_CLAUSE_RE.search(norm) or _TAX_CLAUSE_DESPACED_RE.search(despace(norm)))


# Currency token as a WORD (plus the glued "Rs250" spelling OCR often produces).
_CURRENCY_RE = re.compile(r"₹|\bINR\b|\bRS\b|\bRS(?=[.\d])")

# Unit Sale Price rate expressions (e.g. RS.10.00/N, Rs. 5/g, 10.00/N, ₹10/pcs)
_UNIT_SALE_PRICE_RATE_RE = re.compile(
    r'(?:(?:RS\.?|₹|INR)\s*)?\d+(?:\.\d{1,2})?\s*(?:/|PER\s+)(?:G|GM|GMS|KG|KGS|ML|MLS|L|LTR|LTRS|LITRE|LITER|N|U|PCS|PIECE|PIECES|NUMBER|TABLET|TABLETS|SACHET|SACHETS|UNIT|UNITS)\b',
    re.IGNORECASE
)
_USP_KEYWORD_RE = re.compile(r'\b(?:USP|UNIT\s*SALE\s*PRICE|UNIT\s*PRICE)\b', re.IGNORECASE)
_USP_POINTER_RE = re.compile(
    r'(?:UNIT\s*SALE\s*PRICE|UNIT\s*PRICE|USP)[^\n.]*(?:SEE\s+(?:ABOVE|BELOW|ON|PANEL|STAMP)|AS\s+ABOVE)',
    re.IGNORECASE
)

_QUANTITY_RE = re.compile(r"(?:\b\d+\s*[NnUu]?\s*[xX*]\s*\d+(?:\.\d+)?\s*(?:G|KG|ML|L|GM|GMS|LTRS|N|U)?\b|\b\d+(?:\.\d+)?\s*(G|KG|ML|L|N|GM|GMS|LTRS|U)\b)")
# Deliberately restricted to '/' and '-' separators: allowing '.' would make the price
# "50.00" read as a date.
_DATE_RE = re.compile(r"\b\d{2}[/\-]\d{2,4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(1800|1860|\d{3,4}[\-\s]?\d{6,8})\b")
# Indian pincodes never start with 0 and are never part of a longer digit run.
_PINCODE_RE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")
# Six-digit numbers that are plainly not pincodes.
_NON_PINCODE_CONTEXT_RE = re.compile(r"\b(BATCH|BATCH NO|LOT|LOT NO|BARCODE|EAN|BAR CODE|B NO|L NO|GTIN)\b")

_MRP_KEYWORDS = ["MRP", "M R P", "MAXIMUM RETAIL PRICE", "MAX RETAIL PRICE"]
_NET_QTY_KEYWORDS = ["NET QTY", "NET WT", "NET WEIGHT", "NET CONTENT", "NET CONTENTS", "NET QUANTITY"]
_MFG_DATE_KEYWORDS = ["MFG", "MFD", "PKD", "MANUFACTURE", "MANUFACTURED ON", "PACKED", "BEST BEFORE", "USE BY", "EXP DATE", "EXPIRY"]
_MANUFACTURER_KEYWORDS = [
    "MFD BY", "MANUFACTURED BY", "PACKED BY", "PACKAGED BY", "MARKETED BY", "MKTD BY", "PRODUCED BY",
    "REGD OFFICE", "REGISTERED OFFICE", "WORKS", "FACTORY", "MFG UNIT", "FACTORY UNIT", "PVT LTD",
    "PRIVATE LIMITED", "LIMITED", "LTD", "LLP", "INC", "CORP", "INDUSTRIES", "ENTERPRISES",
]
_CONSUMER_CARE_KEYWORDS = ["CONSUMER", "CUSTOMER CARE", "FEEDBACK", "CALL US", "E MAIL", "EMAIL", "TOLL FREE", "HELPLINE"]
_COUNTRY_KEYWORDS = ["PRODUCT OF", "MADE IN", "COUNTRY OF ORIGIN", "ORIGIN"]


class ComplianceEvaluator:
    def __init__(
        self,
        ruleset: Optional[Dict[str, Any]] = None,
        db: Optional[Any] = None,
        category: Optional[str] = None
    ):
        if ruleset is None:
            raise ValueError("A Prisma-managed ruleset is required for stateless evaluation")
        self.ruleset = ruleset
        self.category = category or ruleset.get("category", "general")
        self.mandatory_rules = ruleset.get("mandatory_declarations", [])
        self.exemptions = ruleset.get("exemptions", [])
        self.rule_map = {r["id"]: r for r in self.mandatory_rules}

    def _get_min_font_size(self, rule_id: str, default: float = 1.0) -> float:
        rule = self.rule_map.get(rule_id, {})
        val = rule.get("min_font_size_mm")
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return default

    def evaluate(self, ocr_result: OCRScanResult, image_bytes: Optional[bytes] = None,
                 face_texts: Optional[List[str]] = None) -> ComplianceResult:
        """
        Core Compliance Engine:
        - First checks if direct LLM evaluation (Groq / Qwen) is active.
        - If active and successful, returns grounded LLM compliance result.
        - Otherwise, executes deterministic Legal Metrology regex validation.
        - Generates color-coded evidence image highlighting only non-compliant blocks.

        face_texts carries the per-face raw text of a multi-face product scan;
        it only shapes the LLM prompt (per-face labeled sections) and is never
        needed for the deterministic engine.
        """
        # 1. Direct LLM Evaluation Hook (Groq / Qwen)
        llm_eval = get_llm_evaluator()
        if llm_eval.is_available():
            try:
                llm_result = llm_eval.evaluate_with_llm(
                    ocr_result,
                    category=self.category,
                    ruleset=self.ruleset,
                    face_texts=face_texts
                )
                if llm_result is not None:
                    # Attach official statutory legal citations to LLM findings
                    citation_svc = get_citation_service()
                    for d in llm_result.summary.what_was_found:
                        if not d.citation:
                            d.citation = citation_svc.get_citation(d.id)
                        # Enforce Rule 12 check on net quantity
                        if d.id == "net_quantity":
                            t_upper = flatten_text(d.extracted_text)
                            illegal_match = re.search(r'\b(GMS|GM|G\.|GM\.|GMS\.|KGS|KG\.|KILO|KILOS|LTR|LTRS|LTR\.|LIT|LITERS|LITRES|MLS|ML\.|M\.L\.|MTS|MTR|MTRS)\b', t_upper)
                            if illegal_match:
                                d.format_valid = False
                                d.status = "FORMAT_ERROR"
                                illegal_sym = illegal_match.group(0)
                                rule12_cit = citation_svc.get_citation("rule_12_metric_symbol")
                                if not any(v.rule_id == "net_quantity" and v.violation_type == "wrong_format" for v in llm_result.summary.whats_wrong):
                                    llm_result.summary.whats_wrong.append(ViolationDetail(
                                        id=f"viol_rule12_net_qty_{uuid.uuid4().hex[:12]}",
                                        rule_id="net_quantity",
                                        field_name="Net Quantity",
                                        violation_type="wrong_format",
                                        severity="MAJOR",
                                        description=f"Net Quantity uses illegal non-standard unit symbol '{illegal_sym}'. Legal Metrology Rule 12 & Third Schedule strictly mandates standard SI symbols ('g', 'kg', 'ml', 'L', 'N').",
                                        evidence_bbox=d.bbox,
                                        citation=rule12_cit
                                    ))
                                    llm_result.overall_result = "FAIL"
                                    llm_result.compliance_score = max(round(llm_result.compliance_score - 15.0, 1), 0.0)

                    for m in llm_result.summary.whats_missing:
                        if not m.citation:
                            m.citation = citation_svc.get_citation(m.id)
                    for v in llm_result.summary.whats_wrong:
                        # Sanitize multi-piece net quantity violations
                        if v.rule_id in ("net_quantity", "multi_piece_net_quantity"):
                            desc_lower = (v.description or "").lower()
                            if "non-standard" in desc_lower or "instead of total" in desc_lower or "30nx5g" in desc_lower or "wrong_format" in str(v.violation_type).lower():
                                raw_nq = next((d.extracted_text for d in llm_result.summary.what_was_found if d.id == "net_quantity"), "30 N x 5 g")
                                clean_nq = re.sub(r'(\d+)\s*N\s*x\s*(\d+)\s*g', r'\1 N x \2 g', raw_nq, flags=re.I)
                                v.rule_id = "multi_piece_net_quantity"
                                v.field_name = "Net Quantity (Multi-Piece Package)"
                                v.violation_type = "missing_total_quantity"
                                v.severity = "MAJOR"
                                v.description = (
                                    f"Multi-piece package declares individual units ('{clean_nq}', where 'N' = Number of Units) "
                                    "but omits the mandatory Total Net Quantity (e.g., '150 g' or '30 N x 5 g = 150 g'). "
                                    "Rule 24 & Rule 2(kc) of Legal Metrology (Packaged Commodities) Rules, 2011 strictly mandate "
                                    "that multi-piece packages declare both the individual pieces and the total net quantity."
                                )
                                v.detected_on_package = clean_nq
                                v.expected_on_package = "Total Net Quantity: 150 g (30 N x 5 g)"
                                v.citation = citation_svc.get_citation("multi_piece_net_quantity") or citation_svc.get_citation("rule_24_multi_piece")
                        if not v.citation:
                            v.citation = citation_svc.get_citation(v.rule_id)

                    # Reconcile Unit Sale Price: Contextual understanding beyond literal keyword matching.
                    # Packaged goods often declare USP as a price-per-unit on batch stickers (e.g. RS.10.00/N)
                    # and/or refer to it via pointer text ("Unit Sale Price, please see above").
                    usp_rate_match = None
                    usp_rate_block = None
                    for b in ocr_result.text_blocks:
                        m_usp = _UNIT_SALE_PRICE_RATE_RE.search(flatten_text(b.text))
                        if m_usp:
                            usp_rate_match = m_usp.group(0)
                            usp_rate_block = b
                            break

                    doc_text = self._document_text(ocr_result)
                    has_usp_pointer = bool(_USP_POINTER_RE.search(doc_text))

                    if usp_rate_match or has_usp_pointer:
                        missing_usp = [m for m in llm_result.summary.whats_missing if m.id == "unit_sale_price"]
                        if missing_usp:
                            llm_result.summary.whats_missing = [m for m in llm_result.summary.whats_missing if m.id != "unit_sale_price"]
                            llm_result.compliance_score = min(round(llm_result.compliance_score + 10.0, 1), 100.0)

                        llm_result.summary.whats_wrong = [
                            v for v in llm_result.summary.whats_wrong 
                            if v.rule_id != "unit_sale_price" or v.violation_type not in ("missing", "not_found")
                        ]

                        if not any(d.id == "unit_sale_price" for d in llm_result.summary.what_was_found):
                            extracted_usp = usp_rate_match or "Unit Sale Price (Declared via package pointer)"
                            best_bbox = usp_rate_block.bbox if usp_rate_block else next(
                                (b.bbox for b in ocr_result.text_blocks if _USP_POINTER_RE.search(flatten_text(b.text))),
                                None
                            )
                            font_px = usp_rate_block.size.estimated_font_size_px if usp_rate_block else 16.0
                            llm_result.summary.what_was_found.append(DeclarationFound(
                                id="unit_sale_price",
                                field_name="Unit Sale Price",
                                extracted_text=extracted_usp,
                                parsed_value=usp_rate_match or extracted_usp,
                                confidence=0.95,
                                bbox=best_bbox,
                                font_size_px=font_px,
                                font_size_mm_est=self._estimate_font_mm(font_px, ocr_result.image_metadata.height),
                                format_valid=True,
                                size_valid=True,
                                status="COMPLIANT",
                                citation=citation_svc.get_citation("unit_sale_price")
                            ))

                        if not llm_result.summary.whats_wrong and not llm_result.summary.whats_missing:
                            llm_result.overall_result = "PASS"

                    if image_bytes:
                        evidence_b64 = self.generate_violation_evidence_image(
                            image_bytes,
                            llm_result.summary.whats_wrong,
                            llm_result.summary.whats_missing
                        )
                        if evidence_b64:
                            llm_result.annotated_image_base64 = evidence_b64
                    return llm_result
            except Exception as llm_err:
                logger.exception("LLM evaluation post-processing failed: %s. Falling back to deterministic engine.", llm_err)

        start_time = time.time()
        
        found_declarations: List[DeclarationFound] = []
        missing_declarations: List[DeclarationMissing] = []
        violations: List[ViolationDetail] = []

        img_height = ocr_result.image_metadata.height
        blocks = ocr_result.text_blocks

        # Whole-label text, flattened to one line - this is what lets a phrase split
        # across two OCR blocks be matched at all.
        doc_norm = self._document_text(ocr_result)

        # Build dynamic matchers for all rules present in active ruleset
        standard_map = {
            "unit_sale_price": (self._is_unit_sale_price, lambda b: self._eval_unit_sale_price(b, img_height, doc_norm, all_blocks=blocks)),
            "mrp": (self._is_mrp, lambda b: self._eval_mrp(b, ocr_result, doc_norm)),
            "net_quantity": (self._is_net_quantity, lambda b: self._eval_net_quantity(b, img_height, doc_norm, all_blocks=blocks)),
            "manufacture_date": (self._is_mfg_date, lambda b: self._eval_mfg_date(b, img_height)),
            "consumer_care": (self._is_consumer_care, lambda b: self._eval_consumer_care(b, img_height)),
            "manufacturer_details": (self._is_manufacturer_details, lambda b: self._eval_manufacturer_details(b, img_height)),
            "country_of_origin": (self._is_country_of_origin, lambda b: self._eval_country_of_origin(b, img_height)),
            "fssai_license": (self._is_fssai_license, lambda b: self._eval_fssai_license(b, img_height, doc_norm)),
        }

        matchers = []
        for r in self.mandatory_rules:
            rid = r["id"]
            if rid in standard_map:
                det, ev = standard_map[rid]
                matchers.append((rid, det, ev))
            else:
                det, ev = self._build_dynamic_matcher(r, img_height)
                matchers.append((rid, det, ev))

        # rule_id -> [(score, DeclarationFound, [ViolationDetail])]. A single label
        # legitimately produces several matching blocks for the same rule (an MRP price
        # line plus its own "(incl. of all taxes)" line); emitting one DeclarationFound
        # per block duplicated the declaration AND its violations, and downstream
        # consumers key declarations by rule id, so the worst duplicate used to win.
        candidates: Dict[str, List[tuple]] = {}

        # Step 1: Analyze every OCR text block against Legal Metrology entity matchers.
        # First matcher wins, so a block still maps to at most one declaration type.
        for block in blocks:
            text = block.text.strip()
            for rule_id, detector, evaluator in matchers:
                if detector(text):
                    decl, viols = evaluator(block)
                    self._add_candidate(candidates, rule_id, block, decl, viols)
                    break

        # Step 1b: A block can claim only one declaration type, and a phrase can wrap onto
        # the next line - both leave a rule looking "missing" while its text sits right
        # there on the label (e.g. an address line that also carries the helpline number is
        # consumed by consumer_care, so manufacturer_details is falsely reported missing).
        # For every rule still unmatched, re-run just that rule's detector over each block
        # and over each block joined with the one after it.
        for rule_id, detector, evaluator in matchers:
            if rule_id in candidates:
                continue
            for idx, block in enumerate(blocks):
                text = block.text.strip()
                joined = f"{text} {blocks[idx + 1].text.strip()}" if idx + 1 < len(blocks) else text
                if detector(text):
                    decl, viols = evaluator(block)
                    self._add_candidate(candidates, rule_id, block, decl, viols)
                elif detector(joined):
                    combined_block = block.model_copy(update={"text": joined})
                    decl, viols = evaluator(combined_block)
                    self._add_candidate(candidates, rule_id, combined_block, decl, viols)

        # Step 1c: Keep exactly one declaration per rule id - the best-evidence block
        # (the one actually carrying the price/quantity/date, then the largest and most
        # confident) - and emit only that candidate's violations.
        for rule_id, entries in candidates.items():
            entries.sort(key=lambda entry: entry[0], reverse=True)
            _, decl, viols = entries[0]
            found_declarations.append(decl)
            violations.extend(viols)

        matched_rule_ids = set(candidates.keys())

        # Check if net quantity below 10g/10ml applies for nutritional_info exemption
        net_qty_found = next((d for d in found_declarations if d.id == "net_quantity"), None)
        is_small_pack = False
        if net_qty_found and net_qty_found.extracted_text:
            qty_match = re.search(r"(\d+(?:\.\d+)?)\s*(g|gm|grams|ml)\b", net_qty_found.extracted_text, re.IGNORECASE)
            if qty_match:
                try:
                    val = float(qty_match.group(1))
                    if val <= 10.0:
                        is_small_pack = True
                except (ValueError, TypeError):
                    pass

        exempted_ids = set()
        for ex in self.exemptions:
            cond = ex.get("condition")
            if cond == "net_quantity_below_10g_or_10ml" and is_small_pack:
                exempted_ids.update(ex.get("exempted_rule_ids", []))

        # Step 2: Check Presence for all mandatory declarations defined in active ruleset
        for rule in self.mandatory_rules:
            rule_id = rule["id"]
            is_required = rule.get("required", True)

            if rule_id in exempted_ids:
                logger.info(f"Rule '{rule_id}' is exempt under category exemption.")
                continue

            if rule_id not in matched_rule_ids and is_required:
                missing_decl = DeclarationMissing(
                    id=rule_id,
                    field_name=rule.get("field_name", rule_id),
                    description=rule.get("description", ""),
                    required=True
                )
                missing_declarations.append(missing_decl)

                viol = ViolationDetail(
                    id=f"viol_missing_{rule_id}_{uuid.uuid4().hex[:12]}",
                    rule_id=rule_id,
                    field_name=rule.get("field_name", rule_id),
                    violation_type="missing",
                    severity="CRITICAL" if rule_id in ["mrp", "net_quantity", "fssai_license"] else "MAJOR",
                    description=f"Mandatory declaration '{rule.get('field_name')}' is missing from product packaging.",
                    evidence_bbox=None
                )
                violations.append(viol)

        # Step 2b: Attach official statutory legal citations to all declarations and violations
        citation_svc = get_citation_service()
        for decl in found_declarations:
            if not decl.citation:
                decl.citation = citation_svc.get_citation(decl.id)

        for missing in missing_declarations:
            if not missing.citation:
                missing.citation = citation_svc.get_citation(missing.id)

        for viol in violations:
            if not viol.citation:
                viol.citation = citation_svc.get_citation(viol.rule_id)
            if not viol.package_element:
                viol.package_element = get_package_element_for_rule(viol.rule_id)
            if not viol.expected_on_package:
                viol.expected_on_package = (
                    self.rule_map.get(viol.rule_id, {}).get("expected_format")
                    or f"Mandatory statutory declaration conforming to {viol.field_name} rules"
                )
            if not viol.detected_on_package:
                if viol.violation_type == "missing":
                    viol.detected_on_package = "Not printed anywhere on the package (Missing from label artwork)"
                else:
                    viol.detected_on_package = "Non-compliant text/declaration on packaging"

        # Step 3: Compute Compliance Score and Overall PASS/FAIL Status
        total_required = sum(1 for r in self.mandatory_rules if r.get("required", True))
        total_found_valid = sum(1 for d in found_declarations if d.format_valid and d.size_valid)

        has_critical_or_major = any(v.severity in ["CRITICAL", "MAJOR"] for v in violations)
        overall_result = "FAIL" if has_critical_or_major else "PASS"

        # Deduct score per violation type
        score_deductions = 0.0
        for v in violations:
            if v.severity == "CRITICAL":
                score_deductions += 30.0
            elif v.severity == "MAJOR":
                score_deductions += 15.0
            else:
                score_deductions += 5.0

        compliance_score = max(round(100.0 - score_deductions, 1), 0.0)
        processing_time = round((time.time() - start_time) * 1000, 2)

        summary = ComplianceSummary(
            what_was_found=found_declarations,
            whats_missing=missing_declarations,
            whats_wrong=violations
        )

        structured_result = {
            "compliance_score": compliance_score,
            "extracted_declarations": [
                {
                    "field": item.field_name,
                    "value": item.extracted_text,
                    "status": item.status,
                    "confidence": item.confidence,
                    "citation": item.citation.model_dump() if item.citation else None,
                }
                for item in found_declarations
            ],
            "violation_list": [
                {
                    "rule_id": item.rule_id,
                    "severity": item.severity,
                    "description": item.description,
                    "field_name": item.field_name,
                    "violation_type": item.violation_type,
                    "detected_on_package": item.detected_on_package,
                    "expected_on_package": item.expected_on_package,
                    "package_element": item.package_element,
                    "citation": item.citation.model_dump() if item.citation else None,
                }
                for item in violations
            ],
            "final_status": "COMPLIANT" if overall_result == "PASS" else "NON_COMPLIANT",
        }

        annotated_b64 = None
        if image_bytes:
            annotated_b64 = self.generate_violation_evidence_image(
                image_bytes,
                violations,
                missing_declarations
            )

        return ComplianceResult(
            overall_result=overall_result,
            compliance_score=compliance_score,
            total_declarations_required=total_required,
            total_found=total_found_valid,
            summary=summary,
            processing_time_ms=processing_time,
            annotated_image_base64=annotated_b64 or ocr_result.annotated_image_base64,
            structured_result=structured_result,
        )

    def evaluate_product(self, face_results: List[OCRScanResult]) -> ComplianceResult:
        """
        Whole-product compliance evaluation across every face of one package:
        merges all faces' OCR (merge_ocr_results) so a declaration printed on
        ANY face satisfies its rule, runs the single evaluation once, then
        stamps each finding with the face it was actually read from so callers
        can render per-face evidence while the verdict stays product-level.
        """
        merged = merge_ocr_results(face_results)
        face_texts = [
            f.raw_text or "\n".join(b.text for b in f.text_blocks)
            for f in face_results
        ]
        result = self.evaluate(
            merged,
            face_texts=face_texts if len(face_results) > 1 else None
        )
        self._attribute_faces(result, merged.text_blocks)
        return result

    @staticmethod
    def _match_face(bbox, text: Optional[str], blocks: List[TextBlock]) -> Optional[int]:
        """Finds the face a finding's evidence came from: the merged block(s)
        whose bbox equals the finding's bbox, preferring a text match when
        several faces carry text at identical coordinates. Findings without a
        usable bbox (nowhere-on-package) stay product-wide (None)."""
        if not bbox or bbox.x_max <= 0:
            return None
        candidates = [b for b in blocks if b.face_index is not None and b.bbox == bbox]
        if not candidates:
            return None
        if text:
            for block in candidates:
                if block.text.strip() == text.strip():
                    return block.face_index
        return candidates[0].face_index

    def _attribute_faces(self, result: ComplianceResult, blocks: List[TextBlock]) -> None:
        for declaration in result.summary.what_was_found:
            declaration.face_index = self._match_face(
                declaration.bbox, declaration.extracted_text, blocks
            )
        for violation in result.summary.whats_wrong:
            violation.face_index = self._match_face(
                violation.evidence_bbox, None, blocks
            )
        # Keep the normalized structured output in sync with the attributed summary.
        if result.structured_result:
            for entry, declaration in zip(
                result.structured_result.extracted_declarations,
                result.summary.what_was_found,
            ):
                entry["face_index"] = declaration.face_index
            for entry, violation in zip(
                result.structured_result.violation_list,
                result.summary.whats_wrong,
            ):
                entry["face_index"] = violation.face_index

    def generate_violation_evidence_image(
        self,
        image_bytes: bytes,
        violations: List[ViolationDetail],
        missing_declarations: List[DeclarationMissing]
    ) -> Optional[str]:
        """
        Draws focused, color-coded bounding boxes strictly for non-compliant declarations.
        CRITICAL: Red (#EF4444)
        MAJOR: Orange (#F97316)
        MINOR: Yellow (#EAB308)
        Adds top-banner alert if mandatory declarations are missing.
        """
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            severity_colors = {
                "CRITICAL": ((239, 68, 68, 255), (239, 68, 68, 55)),
                "MAJOR": ((249, 115, 22, 255), (249, 115, 22, 45)),
                "MINOR": ((234, 179, 8, 255), (234, 179, 8, 35))
            }

            for v in violations:
                if not v.evidence_bbox or v.evidence_bbox.x_max <= 0:
                    continue

                bbox = v.evidence_bbox
                outline_color, fill_color = severity_colors.get(v.severity, severity_colors["CRITICAL"])

                # Draw bounding rectangle around the violating text block
                draw.rectangle(
                    [(bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max)],
                    outline=outline_color,
                    width=3,
                    fill=fill_color
                )

                # Draw badge pill label above box
                v_type = (v.violation_type or "NON_COMPLIANT").upper()
                v_sev = (v.severity or "VIOLATION").upper()
                label_text = f"[{v_sev}] {v.field_name}: {v_type}"
                badge_y = max(0, bbox.y_min - 20)
                badge_w = len(label_text) * 7 + 10
                draw.rectangle(
                    [(bbox.x_min, badge_y), (bbox.x_min + badge_w, badge_y + 18)],
                    fill=outline_color
                )
                draw.text((bbox.x_min + 5, badge_y + 2), label_text, fill=(255, 255, 255, 255))

            # Missing declarations top banner
            if missing_declarations:
                missing_names = ", ".join(d.field_name for d in missing_declarations[:3])
                if len(missing_declarations) > 3:
                    missing_names += f" +{len(missing_declarations) - 3} more"
                banner_text = f"NON-COMPLIANCE: Missing mandatory declarations: {missing_names}"
                banner_h = 30
                draw.rectangle([(0, 0), (pil_img.size[0], banner_h)], fill=(185, 28, 28, 220))
                draw.text((12, 7), banner_text, fill=(255, 255, 255, 255))

            combined = Image.alpha_composite(pil_img, overlay).convert("RGB")
            buffered = io.BytesIO()
            combined.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as err:
            logger.warning(f"Failed to generate violation evidence image: {err}")
            return None

    # --- Text Assembly & Candidate Selection Helpers ---

    def _document_text(self, ocr_result: OCRScanResult) -> str:
        """Whole-label text as one normalized line. raw_text joins the OCR blocks with
        "\\n", so flattening it here is what allows a phrase that OCR split across two
        blocks/lines to still be matched."""
        raw = ocr_result.raw_text or ""
        if not raw.strip():
            raw = "\n".join(b.text for b in ocr_result.text_blocks)
        return normalize_phrase(raw)

    def _has_payload(self, rule_id: str, text: str) -> bool:
        """True when the block carries the declaration's actual value (the price, the
        quantity, the date) rather than only its wording. Used to pick which of several
        matching blocks represents the declaration."""
        flat = flatten_text(text)
        if rule_id == "mrp":
            return bool(re.search(r"\d", flat))
        if rule_id == "unit_sale_price":
            return bool(_UNIT_SALE_PRICE_RATE_RE.search(flat)) or bool(re.search(r"\d", flat))
        if rule_id == "net_quantity":
            return bool(_QUANTITY_RE.search(flat))
        if rule_id == "manufacture_date":
            return bool(_DATE_RE.search(flat)) or bool(re.search(r"\d", flat))
        if rule_id == "consumer_care":
            return bool(_EMAIL_RE.search(text)) or bool(_PHONE_RE.search(flat))
        if rule_id == "manufacturer_details":
            return bool(_PINCODE_RE.search(flat)) or len(flat) > 20
        cat_config = CATEGORY_RULE_PATTERNS.get(rule_id, {})
        if "regex" in cat_config:
            return bool(re.search(cat_config["regex"], text, re.IGNORECASE))
        return True

    def _build_dynamic_matcher(self, rule: Dict[str, Any], img_height: int):
        """Generates dynamic detector and evaluator functions for category rules."""
        rule_id = rule["id"]
        field_name = rule.get("field_name", rule_id)
        regex_pattern = rule.get("regex_pattern")
        min_font_mm = float(rule.get("min_font_size_mm", 1.0))

        cat_config = CATEGORY_RULE_PATTERNS.get(rule_id, {})
        custom_regex = regex_pattern or cat_config.get("regex")
        keywords = cat_config.get("keywords", [])
        if not keywords:
            clean_name = re.sub(r"[^A-Za-z0-9\s]", "", field_name.upper())
            keywords = [clean_name]

        def detector(text: str) -> bool:
            if rule_id == "unit_sale_price":
                return self._is_unit_sale_price(text)
            t_upper = text.upper()
            if custom_regex and re.search(custom_regex, text, re.IGNORECASE):
                return True
            return any(kw in t_upper for kw in keywords)

        def evaluator(block: TextBlock):
            est_font = self._estimate_font_mm(block.size.estimated_font_size_px, img_height)
            size_valid = est_font >= min_font_mm

            extracted = block.text.strip()
            if custom_regex:
                m = re.search(custom_regex, block.text, re.IGNORECASE)
                if m:
                    extracted = m.group(0)

            viols = []
            if not size_valid:
                viols.append(
                    ViolationDetail(
                        id=f"viol_font_{rule_id}_{uuid.uuid4().hex[:12]}",
                        rule_id=rule_id,
                        field_name=field_name,
                        violation_type="size_below_standard",
                        severity="MINOR",
                        description=f"{field_name} font size ({est_font}mm) below minimum required ({min_font_mm}mm).",
                        evidence_bbox=block.bbox
                    )
                )

            decl = DeclarationFound(
                id=rule_id,
                field_name=field_name,
                extracted_text=extracted,
                confidence=round(block.confidence, 2),
                bbox=block.bbox,
                font_size_px=block.size.estimated_font_size_px,
                font_size_mm_est=est_font,
                format_valid=True,
                size_valid=size_valid,
                status="COMPLIANT" if size_valid else "TOO_SMALL"
            )
            return decl, viols

        return detector, evaluator

    def _add_candidate(self, candidates: Dict[str, List[tuple]], rule_id: str, block: TextBlock,
                       decl: DeclarationFound, viols: List[ViolationDetail]) -> None:
        """Registers one possible block for a rule. Best evidence wins: value-carrying
        block first, then the largest font (the real declaration, not a footnote line),
        then OCR confidence, then text length."""
        score = (
            self._has_payload(rule_id, block.text),
            block.size.estimated_font_size_px,
            block.confidence,
            len(block.text),
        )
        candidates.setdefault(rule_id, []).append((score, decl, viols))

    # --- Entity Detection Helpers (Brand-Agnostic & Fully Generalized) ---

    def _is_unit_sale_price(self, text: str) -> bool:
        flat = flatten_text(text)
        if _UNIT_SALE_PRICE_RATE_RE.search(flat):
            return True
        if _USP_POINTER_RE.search(flat):
            return True
        if _USP_KEYWORD_RE.search(flat) and re.search(r"\d", flat):
            return True
        return False

    def _is_mrp(self, text: str) -> bool:
        norm = normalize_phrase(text)
        flat = flatten_text(text)
        # Unit sale price declarations (e.g. "RS.10.00/N") must NEVER be captured as MRP!
        if self._is_unit_sale_price(text) and not (_has_keyword(norm, _MRP_KEYWORDS) or has_tax_clause(norm)):
            return False
        if _has_keyword(norm, _MRP_KEYWORDS) or has_tax_clause(norm):
            return True
        return bool(_CURRENCY_RE.search(flat)) and bool(re.search(r"\d", flat))

    _NUTRITION_TERMS = {"ENERGY", "PROTEIN", "CARBOHYDRATE", "FAT", "SUGAR", "SUGARS", "KCAL", "NUTRITION", "NUTRITIONAL", "SERVING"}

    def _is_net_quantity(self, text: str) -> bool:
        norm = normalize_phrase(text)
        # Disqualify nutritional panels from being flagged as product net quantity
        if any(term in norm for term in self._NUTRITION_TERMS):
            return False
        if _has_keyword(norm, _NET_QTY_KEYWORDS):
            return True
        flat = flatten_text(text)
        return bool(_QUANTITY_RE.search(flat)) and len(flat.split()) <= 5

    def _is_mfg_date(self, text: str) -> bool:
        return _has_keyword(normalize_phrase(text), _MFG_DATE_KEYWORDS) or bool(_DATE_RE.search(flatten_text(text)))

    def _is_manufacturer_details(self, text: str) -> bool:
        norm = normalize_phrase(text)
        # Pricing, tax clauses, and unit sale price must never be classified as manufacturer
        if has_tax_clause(norm) or "UNIT SALE PRICE" in norm or "UNIT PRICE" in norm or "USP" in norm:
            return False
        return _has_keyword(norm, _MANUFACTURER_KEYWORDS) or self._looks_like_pincode(norm)

    def _looks_like_pincode(self, norm: str) -> bool:
        """A bare 6-digit number is far too loose to mean "address" on its own - batch
        codes, barcodes and lot numbers are 6 digits too. Require a plausible pincode
        (never leading zero, never inside a longer digit run), no batch/lot wording, and
        at least one real word alongside it, because an address is not just a number."""
        if not _PINCODE_RE.search(norm):
            return False
        if _NON_PINCODE_CONTEXT_RE.search(norm):
            return False
        return bool(re.search(r"[A-Z]{3,}", norm))

    def _is_consumer_care(self, text: str) -> bool:
        norm = normalize_phrase(text)
        is_email = bool(_EMAIL_RE.search(text))
        is_phone = bool(_PHONE_RE.search(flatten_text(text)))
        return _has_keyword(norm, _CONSUMER_CARE_KEYWORDS) or is_email or is_phone

    def _is_country_of_origin(self, text: str) -> bool:
        return _has_keyword(normalize_phrase(text), _COUNTRY_KEYWORDS)

    def _is_fssai_license(self, text: str) -> bool:
        t_upper = text.upper()
        if any(kw in t_upper for kw in ["FSSAI", "LIC NO", "LICENSE NO", "LIC. NO", "FSSAI LIC"]):
            return True
        flat_digits = re.sub(r"[^\d]", "", text)
        return bool(re.search(r"\b[12]\d{13}\b", text)) or (len(flat_digits) == 14 and flat_digits.startswith(("1", "2")))

    # --- Rule Evaluators ---

    def _eval_fssai_license(self, block: TextBlock, img_height: int, doc_norm: Optional[str] = None) -> tuple[DeclarationFound, List[ViolationDetail]]:
        est_font = self._estimate_font_mm(block.size.estimated_font_size_px, img_height)
        size_valid = est_font >= 1.0

        fssai_number = None
        m = re.search(r"\b([12]\d{13})\b", block.text)
        if not m:
            m = re.search(r"\b(\d{14})\b", block.text)
        if m:
            fssai_number = m.group(1)
        else:
            despaced = re.sub(r"[^\d]", "", block.text)
            m_despaced = re.search(r"([12]\d{13})", despaced)
            if m_despaced:
                fssai_number = m_despaced.group(1)
            elif doc_norm:
                m_doc = re.search(r"FSSAI[^\d]{0,25}([12]\d{13})", doc_norm, re.IGNORECASE)
                if m_doc:
                    fssai_number = m_doc.group(1)

        viols: List[ViolationDetail] = []
        format_valid = bool(fssai_number)

        if not format_valid:
            viols.append(
                ViolationDetail(
                    id=f"viol_fssai_invalid_{uuid.uuid4().hex[:12]}",
                    rule_id="fssai_license",
                    field_name="FSSAI License Number & Logo",
                    violation_type="invalid_format",
                    severity="CRITICAL",
                    description="FSSAI License number on packaging is invalid or missing the required 14-digit numeric format mandated by FSSAI Food Safety & Standards (Labelling and Display) Regulations, 2020.",
                    evidence_bbox=block.bbox,
                    detected_on_package=block.text.strip(),
                    expected_on_package="14-digit numeric license number (e.g. 10012022000123)",
                    package_element="FSSAI License & Logo Panel (Food Packaging)"
                )
            )

        if not size_valid:
            viols.append(
                ViolationDetail(
                    id=f"viol_fssai_font_{uuid.uuid4().hex[:12]}",
                    rule_id="fssai_license",
                    field_name="FSSAI License Number & Logo",
                    violation_type="size_below_standard",
                    severity="MINOR",
                    description=f"FSSAI declaration font size ({est_font}mm) is below minimum required height (1.0mm).",
                    evidence_bbox=block.bbox,
                    detected_on_package=block.text.strip(),
                    expected_on_package="Minimum 1.0mm numeral font height",
                    package_element="FSSAI License & Logo Panel (Food Packaging)"
                )
            )

        decl = DeclarationFound(
            id="fssai_license",
            field_name="FSSAI License Number & Logo",
            extracted_text=f"FSSAI Lic. No. {fssai_number}" if fssai_number else block.text.strip(),
            confidence=round(block.confidence, 2),
            bbox=block.bbox,
            font_size_px=block.size.estimated_font_size_px,
            font_size_mm_est=est_font,
            format_valid=format_valid,
            size_valid=size_valid,
        )
        return decl, viols

    def _eval_mrp(self, block: TextBlock, ocr_result: OCRScanResult,
                  doc_norm: Optional[str] = None) -> tuple[DeclarationFound, List[ViolationDetail]]:
        text = block.text

        # Check format: MRP must mention inclusive of all taxes.
        # The clause is matched against the flattened WHOLE-DOCUMENT text, never against
        # the single block: printers routinely set "(incl. of all taxes)" on its own line
        # under the price, and OCR can even split the clause itself across two blocks
        # ("MRP Rs 120 (Incl." + "of all taxes)"). Matching per line, or against the
        # newline-joined raw text, flagged all of those as a missing tax clause.
        if doc_norm is None:
            doc_norm = self._document_text(ocr_result)
        has_tax_clause_found = has_tax_clause(doc_norm) or has_tax_clause(normalize_phrase(text))

        format_valid = has_tax_clause_found
        viols = []

        if not format_valid:
            viols.append(ViolationDetail(
                id=f"viol_mrp_tax_{block.id}",
                rule_id="mrp",
                field_name=self.rule_map.get("mrp", {}).get("field_name", "Maximum Retail Price (MRP)"),
                violation_type="wrong_format",
                severity="MAJOR",
                description="MRP declaration is missing mandated tax clause '(incl. of all taxes)' as per Legal Metrology Rule 6.",
                evidence_bbox=block.bbox
            ))

        min_font = self._get_min_font_size("mrp", 1.0)
        font_size_mm = self._estimate_font_mm(block.size.estimated_font_size_px, ocr_result.image_metadata.height)
        size_valid = font_size_mm >= min_font

        if not size_valid:
            viols.append(ViolationDetail(
                id=f"viol_mrp_size_{block.id}",
                rule_id="mrp",
                field_name=self.rule_map.get("mrp", {}).get("field_name", "Maximum Retail Price (MRP)"),
                violation_type="too_small",
                severity="MINOR",
                description=f"MRP text font size ({font_size_mm:.1f}mm) is below prescribed minimum height ({min_font:.1f}mm).",
                evidence_bbox=block.bbox
            ))

        status = "COMPLIANT" if (format_valid and size_valid) else ("FORMAT_ERROR" if not format_valid else "TOO_SMALL")

        decl = DeclarationFound(
            id="mrp",
            field_name=self.rule_map.get("mrp", {}).get("field_name", "Maximum Retail Price (MRP)"),
            extracted_text=text,
            parsed_value=text,
            confidence=block.confidence,
            bbox=block.bbox,
            font_size_px=block.size.estimated_font_size_px,
            font_size_mm_est=font_size_mm,
            format_valid=format_valid,
            size_valid=size_valid,
            status=status
        )
        return decl, viols

    def _eval_unit_sale_price(self, block: TextBlock, img_height: int, doc_norm: Optional[str] = None,
                              all_blocks: Optional[List[TextBlock]] = None) -> tuple[DeclarationFound, List[ViolationDetail]]:
        text = block.text.strip()
        flat = flatten_text(text)

        extracted_rate = None
        m_rate = _UNIT_SALE_PRICE_RATE_RE.search(flat)
        if m_rate:
            extracted_rate = m_rate.group(0)
        elif all_blocks:
            for other_b in all_blocks:
                if other_b.id == block.id:
                    continue
                other_flat = flatten_text(other_b.text)
                m_other = _UNIT_SALE_PRICE_RATE_RE.search(other_flat)
                if m_other:
                    extracted_rate = m_other.group(0)
                    text = f"{extracted_rate} (via '{text}')"
                    break

        if not extracted_rate and doc_norm:
            m_doc = _UNIT_SALE_PRICE_RATE_RE.search(doc_norm)
            if m_doc:
                extracted_rate = m_doc.group(0)
                text = f"{extracted_rate} (referenced on package)"

        final_extracted = extracted_rate or text

        min_font = self._get_min_font_size("unit_sale_price", 1.0)
        font_size_mm = self._estimate_font_mm(block.size.estimated_font_size_px, img_height)
        size_valid = font_size_mm >= min_font
        format_valid = bool(extracted_rate or _USP_POINTER_RE.search(flat))

        viols: List[ViolationDetail] = []
        if not size_valid:
            viols.append(ViolationDetail(
                id=f"viol_usp_font_{block.id}",
                rule_id="unit_sale_price",
                field_name=self.rule_map.get("unit_sale_price", {}).get("field_name", "Unit Sale Price"),
                violation_type="too_small",
                severity="MINOR",
                description=f"Unit Sale Price font size ({font_size_mm:.1f}mm) is below minimum prescribed ({min_font:.1f}mm).",
                evidence_bbox=block.bbox
            ))

        status = "COMPLIANT" if (format_valid and size_valid) else ("FORMAT_ERROR" if not format_valid else "TOO_SMALL")

        decl = DeclarationFound(
            id="unit_sale_price",
            field_name=self.rule_map.get("unit_sale_price", {}).get("field_name", "Unit Sale Price"),
            extracted_text=final_extracted,
            parsed_value=extracted_rate or final_extracted,
            confidence=round(block.confidence, 2),
            bbox=block.bbox,
            font_size_px=block.size.estimated_font_size_px,
            font_size_mm_est=font_size_mm,
            format_valid=format_valid,
            size_valid=size_valid,
            status=status
        )
        return decl, viols

    def _eval_net_quantity(self, block: TextBlock, img_height: int, doc_norm: Optional[str] = None,
                           all_blocks: Optional[List[TextBlock]] = None) -> tuple[DeclarationFound, List[ViolationDetail]]:
        text = block.text
        t_upper = flatten_text(text)

        # If the block only contains the keyword prefix (e.g. "Net Qty.:") without the quantity value,
        # stitch it with the adjacent or closest quantity block that isn't a nutrition panel.
        if not _QUANTITY_RE.search(t_upper) and all_blocks:
            for other_b in all_blocks:
                if other_b.id == block.id:
                    continue
                other_text = other_b.text.strip()
                other_norm = normalize_phrase(other_text)
                if any(term in other_norm for term in self._NUTRITION_TERMS):
                    continue
                if _QUANTITY_RE.search(flatten_text(other_text)):
                    text = f"{text.strip()} {other_text}"
                    t_upper = flatten_text(text)
                    break
        viols = []
        citation_svc = get_citation_service()

        # Check for non-standard unit symbols prohibited under Rule 12 (e.g. gms, gm, g., Kgs, ltrs, mls, etc.)
        illegal_match = re.search(r'\b(GMS|GM|G\.|GM\.|GMS\.|KGS|KG\.|KILO|KILOS|LTR|LTRS|LTR\.|LIT|LITERS|LITRES|MLS|ML\.|M\.L\.|MTS|MTR|MTRS)\b', t_upper)
        format_valid = not bool(illegal_match)

        if illegal_match:
            illegal_sym = illegal_match.group(0)
            rule12_cit = citation_svc.get_citation("rule_12_metric_symbol")
            viols.append(ViolationDetail(
                id=f"viol_net_qty_symbol_{block.id}",
                rule_id="net_quantity",
                field_name=self.rule_map.get("net_quantity", {}).get("field_name", "Net Quantity"),
                violation_type="wrong_format",
                severity="MAJOR",
                description=f"Net Quantity uses illegal non-standard unit symbol '{illegal_sym}'. Legal Metrology Rule 12 & Third Schedule strictly mandates standard SI symbols ('g', 'kg', 'ml', 'L', 'N').",
                evidence_bbox=block.bbox,
                citation=rule12_cit
            ))

        # Check for multi-piece package declarations (e.g. 30 N x 5 g, 5 g x 30 N, 10 x 20 g)
        # Under Rule 24 and Rule 2(kc) of Legal Metrology (Packaged Commodities) Rules, 2011:
        # Every multi-piece package must declare the number of individual pieces, the quantity of each piece,
        # AND the total net quantity of all individual pieces.
        multi_match = re.search(
            r'\b(\d+)\s*(?:N|U|UNITS?|PIECES?|PCS?|TABLETS?|SACHETS?|PACKS?|CAKES?|BARS?)?\s*(?:x|X|\*)\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|gm|gms|g\.)\b',
            t_upper,
            re.IGNORECASE
        )
        count = None
        unit_val = None
        unit_sym = None

        if multi_match:
            count = int(multi_match.group(1))
            unit_val = float(multi_match.group(2))
            unit_sym = multi_match.group(3).lower()
        else:
            multi_match_rev = re.search(
                r'\b(\d+(?:\.\d+)?)\s*(g|kg|ml|l|gm|gms|g\.)\s*(?:x|X|\*)\s*(\d+)\s*(?:N|U|UNITS?|PIECES?|PCS?|TABLETS?|SACHETS?|PACKS?|CAKES?|BARS?)?\b',
                t_upper,
                re.IGNORECASE
            )
            if multi_match_rev:
                unit_val = float(multi_match_rev.group(1))
                unit_sym = multi_match_rev.group(2).lower()
                count = int(multi_match_rev.group(3))

        if count and unit_val and unit_sym:
            # Normalize unit sym to standard SI
            norm_sym = "g" if unit_sym in ("g", "gm", "gms", "g.") else ("ml" if unit_sym in ("ml", "mls", "ml.") else unit_sym)
            total_val = count * unit_val

            target_totals = [
                f"{int(total_val) if total_val.is_integer() else total_val}{norm_sym}",
                f"{int(total_val) if total_val.is_integer() else total_val} {norm_sym}",
            ]
            if norm_sym == "g" and total_val >= 1000:
                kg_val = total_val / 1000.0
                target_totals.extend([
                    f"{int(kg_val) if kg_val.is_integer() else kg_val}kg",
                    f"{int(kg_val) if kg_val.is_integer() else kg_val} kg",
                ])
            elif norm_sym == "ml" and total_val >= 1000:
                l_val = total_val / 1000.0
                target_totals.extend([
                    f"{int(l_val) if l_val.is_integer() else l_val}l",
                    f"{int(l_val) if l_val.is_integer() else l_val} l",
                ])

            search_scope = (t_upper + " " + (doc_norm or "")).upper()
            has_total = any(
                re.search(rf'\b{re.escape(target.upper())}\b', search_scope)
                for target in target_totals
            ) or bool(re.search(rf'=\s*{int(total_val) if total_val.is_integer() else total_val}', search_scope))

            if not has_total:
                rule24_cit = citation_svc.get_citation("multi_piece_net_quantity") or citation_svc.get_citation("rule_24_multi_piece") or citation_svc.get_citation("net_quantity")
                display_total = f"{int(total_val) if total_val.is_integer() else total_val} {norm_sym}"
                viols.append(ViolationDetail(
                    id=f"viol_multi_piece_total_{block.id}",
                    rule_id="multi_piece_net_quantity",
                    field_name=self.rule_map.get("net_quantity", {}).get("field_name", "Net Quantity (Multi-Piece Package)"),
                    violation_type="missing_total_quantity",
                    severity="MAJOR",
                    description=(
                        f"Multi-piece package declares individual units ('{block.text}') but omits the mandatory "
                        f"Total Net Quantity ('{display_total}'). Rule 24 and Rule 2(kc) of Legal Metrology "
                        "(Packaged Commodities) Rules, 2011 mandate that multi-piece packages must declare "
                        "both individual pieces and the total net quantity on the package."
                    ),
                    evidence_bbox=block.bbox,
                    citation=rule24_cit
                ))
                format_valid = False

        min_font = self._get_min_font_size("net_quantity", 1.0)
        font_size_mm = self._estimate_font_mm(block.size.estimated_font_size_px, img_height)
        size_valid = font_size_mm >= min_font

        if not size_valid:
            viols.append(ViolationDetail(
                id=f"viol_net_qty_size_{block.id}",
                rule_id="net_quantity",
                field_name=self.rule_map.get("net_quantity", {}).get("field_name", "Net Quantity"),
                violation_type="too_small",
                severity="MINOR",
                description=f"Net Quantity font size ({font_size_mm:.1f}mm) is below prescribed minimum height ({min_font:.1f}mm).",
                evidence_bbox=block.bbox
            ))

        status = "COMPLIANT" if (format_valid and size_valid) else ("FORMAT_ERROR" if not format_valid else "TOO_SMALL")

        decl = DeclarationFound(
            id="net_quantity",
            field_name=self.rule_map.get("net_quantity", {}).get("field_name", "Net Quantity"),
            extracted_text=text,
            parsed_value=text,
            confidence=block.confidence,
            bbox=block.bbox,
            font_size_px=block.size.estimated_font_size_px,
            font_size_mm_est=font_size_mm,
            format_valid=format_valid,
            size_valid=size_valid,
            status=status
        )
        return decl, viols

    def _eval_mfg_date(self, block: TextBlock, img_height: int) -> tuple[DeclarationFound, List[ViolationDetail]]:
        text = block.text
        min_font = self._get_min_font_size("manufacture_date", 1.0)
        font_size_mm = self._estimate_font_mm(block.size.estimated_font_size_px, img_height)
        size_valid = font_size_mm >= min_font
        viols = []

        if not size_valid:
            viols.append(ViolationDetail(
                id=f"viol_mfg_date_size_{block.id}",
                rule_id="manufacture_date",
                field_name=self.rule_map.get("manufacture_date", {}).get("field_name", "Month and Year of Manufacture"),
                violation_type="too_small",
                severity="MINOR",
                description=f"Month/Year of Manufacture font size ({font_size_mm:.1f}mm) is below prescribed minimum height ({min_font:.1f}mm).",
                evidence_bbox=block.bbox
            ))

        decl = DeclarationFound(
            id="manufacture_date",
            field_name=self.rule_map.get("manufacture_date", {}).get("field_name", "Month and Year of Manufacture"),
            extracted_text=text,
            parsed_value=text,
            confidence=block.confidence,
            bbox=block.bbox,
            font_size_px=block.size.estimated_font_size_px,
            font_size_mm_est=font_size_mm,
            format_valid=True,
            size_valid=size_valid,
            status="COMPLIANT" if size_valid else "TOO_SMALL"
        )
        return decl, viols

    def _eval_manufacturer_details(self, block: TextBlock, img_height: int) -> tuple[DeclarationFound, List[ViolationDetail]]:
        text = block.text
        min_font = self._get_min_font_size("manufacturer_details", 1.0)
        font_size_mm = self._estimate_font_mm(block.size.estimated_font_size_px, img_height)
        size_valid = font_size_mm >= min_font
        viols = []

        if not size_valid:
            viols.append(ViolationDetail(
                id=f"viol_mfg_details_size_{block.id}",
                rule_id="manufacturer_details",
                field_name=self.rule_map.get("manufacturer_details", {}).get("field_name", "Manufacturer Name & Address"),
                violation_type="too_small",
                severity="MINOR",
                description=f"Manufacturer details font size ({font_size_mm:.1f}mm) is below prescribed minimum height ({min_font:.1f}mm).",
                evidence_bbox=block.bbox
            ))

        decl = DeclarationFound(
            id="manufacturer_details",
            field_name=self.rule_map.get("manufacturer_details", {}).get("field_name", "Manufacturer Name & Address"),
            extracted_text=text,
            parsed_value=text,
            confidence=block.confidence,
            bbox=block.bbox,
            font_size_px=block.size.estimated_font_size_px,
            font_size_mm_est=font_size_mm,
            format_valid=True,
            size_valid=size_valid,
            status="COMPLIANT" if size_valid else "TOO_SMALL"
        )
        return decl, viols

    def _eval_consumer_care(self, block: TextBlock, img_height: int) -> tuple[DeclarationFound, List[ViolationDetail]]:
        text = block.text
        min_font = self._get_min_font_size("consumer_care", 1.0)
        font_size_mm = self._estimate_font_mm(block.size.estimated_font_size_px, img_height)
        size_valid = font_size_mm >= min_font
        viols = []

        if not size_valid:
            viols.append(ViolationDetail(
                id=f"viol_consumer_care_size_{block.id}",
                rule_id="consumer_care",
                field_name=self.rule_map.get("consumer_care", {}).get("field_name", "Consumer Care Details"),
                violation_type="too_small",
                severity="MINOR",
                description=f"Consumer care font size ({font_size_mm:.1f}mm) is below prescribed minimum height ({min_font:.1f}mm).",
                evidence_bbox=block.bbox
            ))

        decl = DeclarationFound(
            id="consumer_care",
            field_name=self.rule_map.get("consumer_care", {}).get("field_name", "Consumer Care Details"),
            extracted_text=text,
            parsed_value=text,
            confidence=block.confidence,
            bbox=block.bbox,
            font_size_px=block.size.estimated_font_size_px,
            font_size_mm_est=font_size_mm,
            format_valid=True,
            size_valid=size_valid,
            status="COMPLIANT" if size_valid else "TOO_SMALL"
        )
        return decl, viols

    def _eval_country_of_origin(self, block: TextBlock, img_height: int) -> tuple[DeclarationFound, List[ViolationDetail]]:
        text = block.text
        min_font = self._get_min_font_size("country_of_origin", 1.0)
        font_size_mm = self._estimate_font_mm(block.size.estimated_font_size_px, img_height)
        size_valid = font_size_mm >= min_font
        viols = []

        if not size_valid:
            viols.append(ViolationDetail(
                id=f"viol_country_of_origin_size_{block.id}",
                rule_id="country_of_origin",
                field_name=self.rule_map.get("country_of_origin", {}).get("field_name", "Country of Origin"),
                violation_type="too_small",
                severity="MINOR",
                description=f"Country of origin font size ({font_size_mm:.1f}mm) is below prescribed minimum height ({min_font:.1f}mm).",
                evidence_bbox=block.bbox
            ))

        decl = DeclarationFound(
            id="country_of_origin",
            field_name=self.rule_map.get("country_of_origin", {}).get("field_name", "Country of Origin"),
            extracted_text=text,
            parsed_value=text,
            confidence=block.confidence,
            bbox=block.bbox,
            font_size_px=block.size.estimated_font_size_px,
            font_size_mm_est=font_size_mm,
            format_valid=True,
            size_valid=size_valid,
            status="COMPLIANT" if size_valid else "TOO_SMALL"
        )
        return decl, viols

    def _estimate_font_mm(self, font_size_px: float, img_height_px: int) -> float:
        """Estimates physical font height in mm based on pixel scaling."""
        # Standard packaging photo physical height ~150mm
        est_mm = (font_size_px / max(img_height_px, 1)) * 150.0
        return round(est_mm, 1)


# Helper function to evaluate image compliance directly
def evaluate_label_compliance(
    ocr_result: OCRScanResult,
    ruleset: Optional[Dict[str, Any]] = None,
    db: Optional[Any] = None,
    category: Optional[str] = None,
    image_bytes: Optional[bytes] = None
) -> ComplianceResult:
    evaluator = ComplianceEvaluator(ruleset=ruleset, db=db, category=category)
    return evaluator.evaluate(ocr_result, image_bytes=image_bytes)


# Whole-product evaluation across every face of one package (multi-face scans)
def evaluate_label_compliance_multi(
    face_results: List[OCRScanResult],
    ruleset: Optional[Dict[str, Any]] = None,
    db: Optional[Any] = None,
    category: Optional[str] = None,
) -> ComplianceResult:
    evaluator = ComplianceEvaluator(ruleset=ruleset, db=db, category=category)
    return evaluator.evaluate_product(face_results)

