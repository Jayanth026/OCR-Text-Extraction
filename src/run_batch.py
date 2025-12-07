import os
import cv2
from pathlib import Path

from src.preprocessing import preprocess_image
from src.ocr_engine import OCREngine
from src.text_extraction import extract_target_from_ocr
from src.utils import list_images, save_json, draw_highlight


def process_dataset(
    input_folder: str = "tests",
    json_out: str = "results/json",
    screenshot_out: str = "results/screenshots"
):
    """
    Batch-process every image in input_folder.
    Saves:
      - JSON extraction results
      - Highlighted screenshots
    """

    # Ensure output folders exist
    os.makedirs(json_out, exist_ok=True)
    os.makedirs(screenshot_out, exist_ok=True)

    # OCR engine
    ocr = OCREngine()

    # List all image paths
    image_paths = list_images(input_folder)
    print(f"Found {len(image_paths)} images to process.")

    for img_path in image_paths:
        img_name = Path(img_path).name
        print(f"\nProcessing: {img_name}")

        try:
            # Preprocess
            orig, processed = preprocess_image(img_path)

            # OCR
            ocr_lines = ocr.run_ocr(processed)

            # Extract target
            result = extract_target_from_ocr(ocr_lines)

            # Save JSON
            out_json_path = Path(json_out) / f"{img_name}.json"
            save_json({"image_name": img_name, **result}, str(out_json_path))

            # Save highlighted image
            highlighted = draw_highlight(orig, result["target_line"], result["all_lines"])
            out_img_path = Path(screenshot_out) / img_name
            cv2.imwrite(str(out_img_path), highlighted)

            print(f"✓ Saved JSON → {out_json_path}")
            print(f"✓ Saved screenshot → {out_img_path}")

        except Exception as e:
            print(f"✗ Error processing {img_name}: {e}")


if __name__ == "__main__":
    process_dataset()