import easyocr
from typing import List, Dict
import numpy as np

class OCREngine:
    def __init__(self, gpu: bool = False):
        """
        Initialize the EasyOCR reader.
        We use English since the labels are alphanumeric.
        """
        self.reader = easyocr.Reader(['en'], gpu=gpu)

    def run_ocr(self, img: np.ndarray) -> List[Dict]:
        """
        Run OCR on a preprocessed image.
        EasyOCR returns: [
            (bbox, text, confidence),
            ...
        ]
        We convert it to a cleaner dictionary format.
        """
        results = self.reader.readtext(img)

        output = []
        for bbox, text, conf in results:
            output.append({
                "bbox": bbox,
                "text": text,
                "confidence": float(conf)
            })
        return output