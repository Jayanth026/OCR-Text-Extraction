import re
from typing import List, Dict, Optional
import Levenshtein

TARGET_PATTERN = re.compile(r"_1_")


def clean_text(text: str) -> str:
    return text.strip().replace(" ", "")


def is_exact_match(text: str) -> bool:
    return bool(TARGET_PATTERN.search(text))


def fuzzy_contains_pattern(text: str, target: str = "_1_", threshold: float = 0.3) -> bool:
    """
    Check if any substring of the text (same length as target) is similar to the target.
    Works for degraded OCR like '-1-' or '•1•'.
    """
    text = text.strip()
    target_len = len(target)

    if len(text) < target_len:
        return False

    for i in range(len(text) - target_len + 1):
        substring = text[i:i + target_len]
        score = Levenshtein.ratio(substring, target)
        if score >= threshold:
            return True

    return False


def find_target_line(ocr_lines: List[Dict]) -> Optional[Dict]:
    # 1️⃣ Exact match first
    exact_candidates = []
    for line in ocr_lines:
        text = clean_text(line.get("text", ""))
        if is_exact_match(text):
            exact_candidates.append(line)

    if exact_candidates:
        return max(exact_candidates, key=lambda x: x.get("confidence", 0))

    # 2️⃣ Fuzzy match fallback
    fuzzy_candidates = []
    for line in ocr_lines:
        text = clean_text(line.get("text", ""))
        if fuzzy_contains_pattern(text):
            fuzzy_candidates.append(line)

    if fuzzy_candidates:
        return max(fuzzy_candidates, key=lambda x: x.get("confidence", 0))

    return None


def extract_target_from_ocr(ocr_lines: List[Dict]) -> Dict:
    best = find_target_line(ocr_lines)

    if best is None:
        return {
            "target_line": None,
            "confidence": 0.0,
            "all_lines": ocr_lines
        }

    return {
        "target_line": clean_text(best.get("text", "")),
        "confidence": float(best.get("confidence", 0.0)),
        "all_lines": ocr_lines
    }