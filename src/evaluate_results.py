import json
import os
from pathlib import Path


def evaluate_results(json_folder="results/json"):
    json_folder = Path(json_folder)

    print("Looking for JSON files in:", json_folder)

    if not json_folder.exists():
        print("❌ Folder does not exist!")
        return

    files = list(json_folder.glob("*.json"))
    print("Found JSON files:", len(files))

    if len(files) == 0:
        print("❌ No JSON files found in results/json")
        return

    extracted = 0
    missing = 0

    for jf in files:
        with open(jf, "r") as f:
            data = json.load(f)

        target = data.get("target_line")

        if target and "_1_" in target:
            extracted += 1
        else:
            missing += 1

    print("\n===== Accuracy Summary =====")
    print(f"Total JSON files: {len(files)}")
    print(f"Extracted: {extracted}")
    print(f"Missing: {missing}")
    print(f"Accuracy: {(extracted / len(files)) * 100:.2f}%")
    print("============================\n")


if __name__ == "__main__":
    evaluate_results()