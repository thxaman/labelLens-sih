import os
import re
import json
import time
import uuid
import logging
from typing import Optional, Dict, Any, List
import httpx

from schemas.ocr import OCRScanResult, TextBlock, BBox
from schemas.compliance import (
    ComplianceResult,
    ComplianceSummary,
    DeclarationFound,
    DeclarationMissing,
    ViolationDetail,
    StructuredComplianceResult,
)
from schemas.llm_compliance import LLMRuleEvaluation, LLMComplianceResponse
from services.rag.citation_service import get_citation_service

logger = logging.getLogger("llm_evaluator")


def normalize_for_grounding(text: str) -> str:
    """Strips punctuation, extra spaces, and lowercases text for fuzzy grounding match."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text or "").lower()
    return " ".join(cleaned.split())


def verify_grounding(exact_quote: Optional[str], raw_text: str) -> bool:
    """
    Verifies that the LLM's cited exact_quote actually exists in the OCR text.
    Prevents hallucinations from fabricating declarations not present on packaging.
    """
    if not exact_quote or not exact_quote.strip():
        return False

    norm_quote = normalize_for_grounding(exact_quote)
    norm_doc = normalize_for_grounding(raw_text)

    if not norm_quote:
        return False

    # Exact substring check
    if norm_quote in norm_doc:
        return True

    # Despaced substring check (for glued OCR tokens like Rs250 or 100g)
    despaced_quote = norm_quote.replace(" ", "")
    despaced_doc = norm_doc.replace(" ", "")
    if despaced_quote in despaced_doc:
        return True

    # Word-level overlap: if >= 75% of quote words appear in document
    words = norm_quote.split()
    if len(words) >= 3:
        matches = sum(1 for w in words if w in norm_doc)
        if matches / len(words) >= 0.75:
            return True

    return False


def find_matching_block(exact_quote: Optional[str], text_blocks: List[TextBlock]) -> Optional[TextBlock]:
    """Finds the OCR text block that best matches the exact_quote."""
    if not exact_quote or not text_blocks:
        return None

    norm_quote = normalize_for_grounding(exact_quote)
    best_match_block = None
    best_score = 0.0

    for block in text_blocks:
        norm_block = normalize_for_grounding(block.text)
        if norm_quote in norm_block or norm_block in norm_quote:
            return block

        # Token overlap score
        q_words = set(norm_quote.split())
        b_words = set(norm_block.split())
        if q_words and b_words:
            overlap = len(q_words & b_words) / float(len(q_words))
            if overlap > best_score and overlap >= 0.5:
                best_score = overlap
                best_match_block = block

    return best_match_block


def find_matching_bbox(exact_quote: Optional[str], text_blocks: List[TextBlock]) -> Optional[BBox]:
    """Finds the bounding box of the OCR text block that best matches the exact_quote."""
    block = find_matching_block(exact_quote, text_blocks)
    return block.bbox if block else None


PACKAGE_ELEMENT_MAP = {
    "net_quantity": "Principal Display Panel (PDP) - Net Quantity Declaration",
    "net_quantity_format": "Principal Display Panel (PDP) - Metric Notation",
    "net_quantity_size": "Principal Display Panel (PDP) - Numeral Height",
    "mrp": "Pricing & MRP Box (Principal Display Panel)",
    "unit_sale_price": "Unit Sale Price Panel (Next to MRP)",
    "manufacture_date": "Date Coding & Batch Stamp",
    "mfg_date": "Date Coding & Batch Stamp",
    "manufacturer_details": "Manufacturer / Packer / Importer Address Block",
    "packer_details": "Packer / Importer Address Block",
    "consumer_care": "Consumer Care & Grievance Helpline Box",
    "country_of_origin": "Country of Origin Declaration",
    "fssai_license": "FSSAI License & Logo Panel (Food Packaging)",
    "veg_nonveg_symbol": "Vegetarian / Non-Vegetarian Emblem (PDP)",
    "nutritional_info": "Nutritional Facts Table (Back of Pack / Side Panel)",
    "ingredients_list": "Ingredients Statement (Descending Weight/Volume)",
    "allergen_info": "Allergen Callout & Warning Notice",
    "mfg_license": "Drug & Cosmetic Manufacturing License Block",
    "batch_number": "Batch / Lot Identification Stamp",
    "directions_for_use": "Directions for Safe Use & Application Panel",
    "cosmetic_warnings": "Precautionary & Warning Statement",
    "fibre_composition": "Fibre Composition % Tag (Finished Garments)",
    "size_declaration": "Garment Size Indicator & Metric Dimensions",
    "wash_care": "Wash & Care Instructions Label",
    "bis_registration": "BIS Standard Mark & CRS Registration Plate",
    "power_ratings": "Electrical Specification & Voltage Rating",
    "pan_masala_warning": "Statutory Health Warning Notice (Front of Pack)",
    "pan_masala_no_exemption": "Net Quantity & Statutory Package Declaration",
    "qr_code_declaration": "Digital E-Label / QR Code Panel",
}

def get_package_element_for_rule(rule_id: str) -> str:
    return PACKAGE_ELEMENT_MAP.get(rule_id.lower().strip(), "Principal Display Panel (PDP)")


class LLMComplianceEvaluator:
    def __init__(self):
        # Support Groq API key directly or generic LLM_API_KEY
        self.api_key = (
            os.environ.get("GROQ_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or ""
        ).strip()
        # Require explicit LLM_BACKEND="groq" or "ollama" to activate remote LLM evaluation;
        # otherwise use the high-speed deterministic compliance engine with statutory citations.
        self.backend = os.environ.get("LLM_BACKEND", "none").strip().lower()

        # Groq OpenAI-compatible endpoint
        self.base_url = (
            os.environ.get("GROQ_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or "https://api.groq.com/openai/v1"
        ).rstrip("/")

        # Model defaults to Qwen or user override
        self.model = (
            os.environ.get("GROQ_MODEL")
            or os.environ.get("LLM_MODEL")
            or "qwen-2.5-32b"
        )
        self.timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "15.0"))

    def is_available(self) -> bool:
        """Returns True if LLM backend is configured with a valid API key."""
        if self.backend in ("none", "", "disabled"):
            return False
        if not self.api_key:
            return False
        return True

    def _build_prompt(
        self,
        ocr_result: OCRScanResult,
        category: str,
        ruleset: Dict[str, Any],
        face_texts: Optional[List[str]] = None
    ) -> tuple[str, str]:
        system_prompt = (
            "You are an expert Legal Metrology Compliance Inspector for packaged commodities in India.\n"
            "Evaluate the provided OCR label text against India's Legal Metrology (Packaged Commodities) Rules, 2011 "
            "and applicable category-specific regulations.\n\n"
            "CRITICAL RULES:\n"
            "1. Evaluate based ONLY on verbatim text from OCR scan.\n"
            "2. For every rule marked 'PASS', copy the exact verbatim text into 'exact_quote'. Do NOT invent text.\n"
            "3. If a mandatory declaration is missing, mark status='FAIL', violation_type='missing', exact_quote=null, detected_on_package='Not printed on package (Missing from label)'.\n"
            "4. For EVERY violation, explicitly report:\n"
            "   - 'detected_on_package': what exact text/declaration is printed on the package (or 'Not printed on package')\n"
            "   - 'expected_on_package': what the package is legally mandated to display instead\n"
            "   - 'package_element': what specific area or component of the package packaging is in violation\n"
            "5. Net Quantity & Multi-Piece Package Rules (Rule 13 & Rule 24):\n"
            "   - Standard SI units are: 'g', 'kg', 'ml', 'L', and 'N' or 'U' (where 'N' or 'U' is the STATUTORY symbol for Number/Count/Units under Rule 13(5)(ii)). 'N' is 100% legal for piece count!\n"
            "   - Multi-piece packages (e.g. '30 N x 5 g' or '10 x 20 g'):\n"
            "     * Declaring the piece count ('30 N') and unit weight ('5 g') is legally valid for individual pieces.\n"
            "     * However, Rule 24 and Rule 2(kc) mandate that multi-piece packages MUST also declare the Total Net Quantity (e.g. '150 g' or '30 N x 5 g = 150 g').\n"
            "     * If total net weight is missing, do NOT call 'N' non-standard! Mark violation_type='missing_total_quantity', explain: 'Multi-piece package declares individual units (30 N x 5 g) but omits mandatory Total Net Quantity (150 g) under Rule 24'.\n"
            "   - Prohibited non-standard symbols are: 'gms', 'gm', 'ltrs', 'kgs'. If used, mark violation_type='wrong_format'.\n"
            "6. Unit Sale Price (USP) Recognition — VERY IMPORTANT:\n"
            "   - A Unit Sale Price is ANY price-per-unit expression in the format: PRICE/UNIT or PRICE PER UNIT.\n"
            "   - Valid unit denominators include: /N, /U, /g, /kg, /ml, /L, /pcs, /piece, /tablet, /sachet, /pack, /unit.\n"
            "   - Examples that ARE a valid Unit Sale Price: 'RS.10.00/N', 'Rs.10/N', 'Rs 5.00/g', '₹10/pcs', 'MRP Rs.10.00/N', 'RS.9.00/N'.\n"
            "   - The label text 'Unit Sale Price, please see above' or 'Unit Sale Price as above' means the USP IS declared on another sticker/panel on the same package — treat this as PASS with exact_quote='(see sticker above)'.\n"
            "   - Do NOT mark USP as missing if ANY price-per-unit expression (e.g. RS.10.00/N) appears ANYWHERE in the OCR text.\n"
            "7. Always preserve exact verbatim spacing and capitalization from OCR (e.g. '30 N x 5 g', never squish to '30Nx5g').\n"
            "8. Keep 'explanation' concise (maximum 15 words).\n"
            "9. Return ONLY a valid JSON object matching the requested schema without any markdown formatting.\n"
            "10. MULTI-FACE PRODUCT EVALUATION (applies when the OCR text contains several label faces of the SAME product):\n"
            "   - A mandatory declaration printed on ANY face is PRESENT on the product.\n"
            "   - Mark status='FAIL' with violation_type='missing' ONLY when the declaration is absent from ALL faces.\n"
            "   - Cite the exact_quote from the face where the declaration actually appears.\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "category": "string",\n'
            '  "overall_result": "PASS" | "FAIL",\n'
            '  "compliance_score": number (0-100),\n'
            '  "summary": "concise inspection summary",\n'
            '  "evaluations": [\n'
            '    {\n'
            '      "rule_id": "string",\n'
            '      "status": "PASS" | "FAIL" | "EXEMPT",\n'
            '      "extracted_value": "parsed value string or null",\n'
            '      "exact_quote": "exact verbatim substring from OCR text or null",\n'
            '      "detected_on_package": "what was printed on the package (e.g. Net Qty: 30 N x 5 g)",\n'
            '      "expected_on_package": "what the package must display (e.g. Total Net Quantity: 150 g (30 N x 5 g))",\n'
            '      "package_element": "package area in violation (e.g. Principal Display Panel - Net Quantity)",\n'
            '      "violation_type": "missing" | "wrong_format" | "too_small" | "missing_total_quantity" | null,\n'
            '      "severity": "CRITICAL" | "MAJOR" | "MINOR" | null,\n'
            '      "explanation": "concise rationale for finding"\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        mandatory_rules = ruleset.get("mandatory_declarations", [])
        rules_desc = []
        for r in mandatory_rules:
            rules_desc.append(
                f"- Rule ID: '{r['id']}' | Name: '{r['field_name']}' | Required Format: {r.get('expected_format', '')}"
            )

        exemptions = ruleset.get("exemptions", [])
        exempt_desc = []
        for ex in exemptions:
            exempt_desc.append(f"- Exemption for '{ex.get('rule_id')}': {ex.get('description', '')}")

        user_content = (
            f"Product Category: {category}\n\n"
            f"Mandatory Rules to Evaluate:\n" + "\n".join(rules_desc) + "\n\n"
        )
        if exempt_desc:
            user_content += "Statutory Exemptions:\n" + "\n".join(exempt_desc) + "\n\n"

        if face_texts and len(face_texts) > 1:
            sections = "\n\n".join(
                f"--- Face {i + 1} ---\n{text}" for i, text in enumerate(face_texts)
            )
            user_content += (
                f"OCR Extracted Packaging Text ({len(face_texts)} label faces of the SAME product):\n"
                f"'''\n{sections}\n'''\n\n"
                "Treat all faces as one product when assessing compliance and output JSON."
            )
        else:
            user_content += (
                f"OCR Extracted Packaging Text:\n"
                f"'''\n{ocr_result.raw_text}\n'''\n\n"
                "Perform legal metrology compliance assessment and output JSON."
            )

        return system_prompt, user_content

    def evaluate_with_llm(
        self,
        ocr_result: OCRScanResult,
        category: str,
        ruleset: Dict[str, Any],
        face_texts: Optional[List[str]] = None
    ) -> Optional[ComplianceResult]:
        """
        Runs direct LLM compliance evaluation via Groq with anti-hallucination grounding.
        face_texts carries the per-face raw text of a multi-face product scan so the
        prompt can show the model which face each declaration sits on.
        Returns ComplianceResult if successful, or None to fall back to the deterministic regex engine.
        """
        if not self.is_available():
            return None

        start_time = time.time()
        system_prompt, user_content = self._build_prompt(ocr_result, category, ruleset, face_texts=face_texts)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "600"))
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                # Rate limit retry with 3.5s backoff if 429 occurs
                if resp.status_code == 429:
                    logger.warning("Groq rate limit (429) encountered. Pausing 3.5s before retry...")
                    time.sleep(3.5)
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    )

            if resp.status_code != 200:
                logger.warning(
                    "Groq LLM evaluation returned status %d: %s. Falling back to deterministic engine.",
                    resp.status_code,
                    resp.text[:200]
                )
                return None

            data = resp.json()
            raw_response_text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            if not raw_response_text:
                logger.warning(
                    "Groq returned an empty model output (finish_reason=%s). Falling back to deterministic engine.",
                    data.get("choices", [{}])[0].get("finish_reason", "unknown")
                )
                return None
            # Clean markdown JSON block formatting if present
            if raw_response_text.startswith("```"):
                raw_response_text = re.sub(r"^```(?:json)?\s*", "", raw_response_text)
                raw_response_text = re.sub(r"\s*```$", "", raw_response_text)

            parsed = json.loads(raw_response_text)
            llm_resp = LLMComplianceResponse(**parsed)

            # Grounding verification, citation attachment, and ComplianceResult mapping
            citation_svc = get_citation_service()
            rule_map = {r["id"]: r for r in ruleset.get("mandatory_declarations", [])}
            found_declarations: List[DeclarationFound] = []
            missing_declarations: List[DeclarationMissing] = []
            violations: List[ViolationDetail] = []

            for ev in llm_resp.evaluations:
                rid = ev.rule_id
                rule_meta = rule_map.get(rid, {"field_name": rid})
                field_name = rule_meta.get("field_name", rid)
                citation = citation_svc.get_citation(rid)

                # Anti-hallucination verification
                if ev.status == "PASS" and ev.exact_quote:
                    is_grounded = verify_grounding(ev.exact_quote, ocr_result.raw_text)
                    ev.grounded = is_grounded
                    if not is_grounded:
                        logger.warning(
                            "LLM claimed rule '%s' PASS but exact_quote '%s' was NOT grounded in OCR text. Downgrading to FAIL.",
                            rid,
                            ev.exact_quote
                        )
                        ev.status = "FAIL"
                        ev.violation_type = "missing"
                        ev.severity = "CRITICAL"
                        ev.explanation = f"Declaration was not verifiable in actual label text ({ev.explanation})"

                matched_block = find_matching_block(ev.exact_quote, ocr_result.text_blocks)
                bbox = matched_block.bbox if matched_block else BBox(
                    x_min=0, y_min=0, x_max=0, y_max=0
                )

                if ev.status == "PASS":
                    confidence = round(matched_block.confidence, 2) if (matched_block and matched_block.confidence) else 0.95
                    font_size_px = (
                        matched_block.size.estimated_font_size_px
                        if (matched_block and hasattr(matched_block, "size") and matched_block.size and matched_block.size.estimated_font_size_px)
                        else 20.0
                    )
                    img_h = ocr_result.image_metadata.height if (ocr_result and ocr_result.image_metadata and ocr_result.image_metadata.height) else 1000
                    font_size_mm_est = round(max((font_size_px / max(img_h, 1)) * 150.0, 1.0), 1)

                    found_declarations.append(
                        DeclarationFound(
                            id=rid,
                            field_name=field_name,
                            extracted_text=ev.exact_quote or ev.extracted_value or "",
                            parsed_value=ev.extracted_value,
                            confidence=confidence,
                            bbox=bbox,
                            font_size_px=font_size_px,
                            font_size_mm_est=font_size_mm_est,
                            format_valid=True,
                            size_valid=True,
                            status="COMPLIANT",
                            citation=citation
                        )
                    )
                elif ev.status == "EXEMPT":
                    pass
                else:
                    # FAIL
                    if ev.violation_type == "missing":
                        missing_declarations.append(
                            DeclarationMissing(
                                id=rid,
                                field_name=field_name,
                                description=rule_meta.get("description", ""),
                                required=rule_meta.get("required", True),
                                citation=citation
                            )
                        )
                    else:
                        confidence = round(matched_block.confidence, 2) if (matched_block and matched_block.confidence) else 0.95
                        font_size_px = (
                            matched_block.size.estimated_font_size_px
                            if (matched_block and hasattr(matched_block, "size") and matched_block.size and matched_block.size.estimated_font_size_px)
                            else 20.0
                        )
                        img_h = ocr_result.image_metadata.height if (ocr_result and ocr_result.image_metadata and ocr_result.image_metadata.height) else 1000
                        font_size_mm_est = round(max((font_size_px / max(img_h, 1)) * 150.0, 1.0), 1)
                        found_declarations.append(
                            DeclarationFound(
                                id=rid,
                                field_name=field_name,
                                extracted_text=ev.exact_quote or ev.detected_on_package or ev.extracted_value or "",
                                parsed_value=ev.extracted_value or ev.detected_on_package,
                                confidence=confidence,
                                bbox=bbox,
                                font_size_px=font_size_px,
                                font_size_mm_est=font_size_mm_est,
                                format_valid=False,
                                size_valid=ev.violation_type != "too_small",
                                status="FAIL",
                                citation=citation
                            )
                        )
                    package_elem = (
                        ev.package_element
                        or get_package_element_for_rule(rid)
                    )
                    expected_val = (
                        ev.expected_on_package
                        or rule_meta.get("expected_format")
                        or f"Mandatory statutory declaration conforming to {rid.replace('_', ' ').title()} rules"
                    )
                    if ev.detected_on_package:
                        detected_val = ev.detected_on_package
                    elif ev.violation_type == "missing":
                        detected_val = "Not printed on package (Missing from label artwork)"
                    elif ev.exact_quote:
                        detected_val = f"'{ev.exact_quote}'"
                    elif ev.extracted_value:
                        detected_val = f"'{ev.extracted_value}'"
                    else:
                        detected_val = "Non-compliant declaration on label"

                    violations.append(
                        ViolationDetail(
                            id=f"viol_llm_{rid}_{uuid.uuid4().hex[:12]}",
                            rule_id=rid,
                            field_name=field_name,
                            violation_type=ev.violation_type or "missing",
                            severity=ev.severity or "CRITICAL",
                            description=ev.explanation,
                            detected_on_package=detected_val,
                            expected_on_package=expected_val,
                            package_element=package_elem,
                            evidence_bbox=bbox if (bbox.x_max > 0) else None,
                            citation=citation
                        )
                    )

            total_required = len(rule_map)
            total_found_valid = len(found_declarations)
            overall_result = "PASS" if (len(violations) == 0 and len(missing_declarations) == 0) else "FAIL"
            score = llm_resp.compliance_score if llm_resp.compliance_score is not None else (
                round((total_found_valid / max(total_required, 1)) * 100.0, 1)
            )
            processing_time = round((time.time() - start_time) * 1000, 2)

            summary = ComplianceSummary(
                what_was_found=found_declarations,
                whats_missing=missing_declarations,
                whats_wrong=violations
            )

            structured_result = StructuredComplianceResult(
                compliance_score=score,
                extracted_declarations=[
                    {
                        "field": item.field_name,
                        "value": item.extracted_text,
                        "status": item.status,
                        "confidence": item.confidence,
                        "citation": item.citation.model_dump() if item.citation else None,
                    }
                    for item in found_declarations
                ],
                violation_list=[
                    {
                        "rule_id": item.rule_id,
                        "severity": item.severity,
                        "description": item.description,
                        "field_name": item.field_name,
                        "violation_type": item.violation_type,
                        "citation": item.citation.model_dump() if item.citation else None,
                    }
                    for item in violations
                ],
                final_status="COMPLIANT" if overall_result == "PASS" else "NON_COMPLIANT"
            )

            return ComplianceResult(
                overall_result=overall_result,
                compliance_score=score,
                total_declarations_required=total_required,
                total_found=total_found_valid,
                summary=summary,
                processing_time_ms=processing_time,
                annotated_image_base64=ocr_result.annotated_image_base64,
                structured_result=structured_result
            )

        except Exception as e:
            logger.error("Groq LLM evaluation encountered exception: %s. Falling back to regex.", e)
            return None


# Module singleton
_llm_evaluator_instance: Optional[LLMComplianceEvaluator] = None

def get_llm_evaluator() -> LLMComplianceEvaluator:
    global _llm_evaluator_instance
    if _llm_evaluator_instance is None:
        _llm_evaluator_instance = LLMComplianceEvaluator()
    return _llm_evaluator_instance
