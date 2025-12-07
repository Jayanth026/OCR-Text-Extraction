# OCR Shipping Label Extractor — `_1_` Target Line Detection

This project implements an OCR-based pipeline to extract the complete text line containing the pattern `_1_` from shipping label / waybill images. It includes preprocessing, OCR inference, fuzzy substring matching, batch processing, accuracy reporting, automated tests, and a Streamlit demo app.

---

## Features

- Robust preprocessing (resize → grayscale → denoise → threshold)
- EasyOCR-based text extraction
- Custom fuzzy matching using Levenshtein distance
- Batch inference for all test images
- JSON output and highlighted screenshots for each image
- Streamlit web interface for interactive testing
- Automated unit tests for pipeline components
- Accuracy evaluation script (src/evaluate_results.py)

---

## Project Structure

```
project-root/
├── app.py                     # Streamlit UI
├── requirements.txt
├── README.md
├── results/
│   ├── json/                  # OCR outputs
│   ├── screenshots/           # Highlighted label images
│   └── accuracy_report.md     # Accuracy summary
├── src/
│   ├── preprocessing.py       # Preprocessing pipeline
│   ├── ocr_engine.py          # EasyOCR wrapper
│   ├── text_extraction.py     # Core extraction logic
│   ├── utils.py               # JSON saving + image annotation
│   ├── run_batch.py           # Batch inference script
│   └── evaluate_results.py    # Accuracy evaluator
├── tests/                     # Unit tests
└── notebooks/                 # Experimentation notebooks
```

---

## Installation

### 1. Clone the repository

```
git clone <your-repo-url>
cd ocr-shipping-label-1-line
```

### 2. Create a virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Run the Streamlit App

```
streamlit run app.py
```

Upload an image and view:
- Preprocessed version
- OCR output
- Extracted `_1_` target line
- Confidence score
- Highlighted detected line

---

### Run Batch Processing

```
python -m src.run_batch
```

Outputs are stored in:

```
results/json/
results/screenshots/
```

---

### Run Accuracy Evaluation

```
python -m src.evaluate_results
```

Example output:

```
Looking for JSON files in: results/json
Found JSON files: 27

===== Accuracy Summary =====
Total JSON files: 27
Extracted: 3
Missing: 24
Accuracy: 11.11%
```

---

## Technical Approach

### 1. Preprocessing

Each image undergoes:
- Resize (max dimension 1500px)
- Grayscale conversion
- Gaussian denoising
- Adaptive thresholding

This improves OCR performance on noisy shipping labels.

---

### 2. OCR Engine

- Built on EasyOCR
- Returns bounding boxes, text, and confidence scores
- Cached in Streamlit for fast UI performance

---

### 3. Text Extraction

#### Exact Pattern Match
Detects `_1_` directly inside text.

#### Fuzzy Matching
Handles degraded OCR patterns such as:
- `-1-`
- `.1.`
- `_1-`
- missing underscores

Uses:
- Levenshtein distance
- Sliding-window substring comparison
- Confidence-based ranking

---

## Accuracy Summary

Accuracy is computed by scanning all JSON outputs and checking whether the extracted text contains `_1_`.

Example:

```
Total images processed: 27
Extracted target line: 3
Missing predictions: 24
Accuracy: 11.11%
```

More detailed notes are available in `results/accuracy_report.md`.

---

## Unit Tests

Run tests:

```
python -m pytest tests -q
```

Tests include:
- Preprocessing validation
- Fuzzy extraction logic tests
- JSON serialization checks

---

## Challenges and Solutions

### Misread underscores
OCR frequently confuses `_` with `-`.

Solution: fuzzy substring matching with Levenshtein.

### Non-serializable OCR outputs
EasyOCR returns numpy types (`np.int32`, `np.float64`).

Solution: recursive numpy-to-Python conversion in utils.py.

### Low-quality label images
Labels often contain noise, shadows, and distortion.

Solution: resizing, denoising, and adaptive threshold preprocessing.

---

## Future Improvements

- Try PaddleOCR or TrOCR for improved recognition
- Add deskewing and perspective correction
- Dynamic fuzzy thresholding based on OCR confidence
- Train a custom CRNN model
- Enhance pattern validation with regex scoring

---

## Submission Notes

- JSON outputs in `results/json/`
- Screenshots in `results/screenshots/`
- Accuracy report in `results/accuracy_report.md`
- Streamlit app included
- Unit tests included
- WhatsApp notifications sent as required

---

## End of README