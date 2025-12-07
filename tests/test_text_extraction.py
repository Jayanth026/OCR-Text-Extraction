from src.text_extraction import extract_target_from_ocr

def test_extract_target_exact_match():
    fake_ocr = [
        {"text": "ABCDEF_1_GHI", "confidence": 0.95, "bbox": []},
        {"text": "Other text", "confidence": 0.80, "bbox": []},
    ]

    result = extract_target_from_ocr(fake_ocr)

    assert result["target_line"] == "ABCDEF_1_GHI"
    assert result["confidence"] == 0.95


def test_extract_target_fuzzy_match():
    fake_ocr = [
        {"text": "ABCDEF-1-GHI", "confidence": 0.90, "bbox": []},  # '-' instead of '_'
    ]

    result = extract_target_from_ocr(fake_ocr)

    assert result["target_line"] is not None  # Fuzzy logic should catch it