import os
import json
from typing import List, Dict, Set

import cv2
import numpy as np


def list_images(folder: str, exts: Set[str] = {".jpg", ".jpeg", ".png"}) -> List[str]:
    """
    List all image files in a folder with allowed extensions.
    """
    paths = []
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname.lower())[1]
        if ext in exts:
            paths.append(os.path.join(folder, fname))
    paths.sort()
    return paths


def to_python_type(obj):
    """
    Recursively convert numpy types and arrays into Python-native types.
    """
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj


def save_json(data: Dict, path: str) -> None:
    """
    Save a dictionary as pretty-printed JSON, converting numpy types.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean_data = to_python_type(data)
    with open(path, "w") as f:
        json.dump(clean_data, f, indent=4)


def draw_highlight(image: np.ndarray, target_text: str, ocr_lines: List[Dict]) -> np.ndarray:
    """
    Draw a green polygon around the OCR line that contains target_text.
    If multiple lines contain it, highlight them all.
    """
    output = image.copy()

    if not target_text:
        return output

    for line in ocr_lines:
        text = line.get("text", "")
        if target_text and target_text in text:
            bbox = line.get("bbox")
            if bbox is None:
                continue

            pts = np.array(bbox, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))

            cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    return output