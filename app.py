import streamlit as st
import numpy as np
import cv2

from src.preprocessing import preprocess_image_array
from src.ocr_engine import OCREngine
from src.text_extraction import extract_target_from_ocr
from src.utils import draw_highlight


# Streamlit Title
st.title("🔍 Shipping Label OCR — Extract '_1_' Line")
st.write("Upload a waybill/shipping label image and extract the full text containing the pattern `_1_`.")


# Initialize OCR engine once
@st.cache_resource
def load_ocr_engine():
    return OCREngine()

ocr = load_ocr_engine()


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("📷 Uploaded Image")
    st.image(image, channels="BGR", use_column_width=True)

    # Preprocess
    orig, processed = preprocess_image_array(image)

    st.subheader("⚙️ Preprocessed Image")
    st.image(processed, caption="Preprocessed for OCR", use_column_width=True)

    # Run OCR
    with st.spinner("Running OCR..."):
        ocr_lines = ocr.run_ocr(processed)
        result = extract_target_from_ocr(ocr_lines)

    st.subheader("🔎 Extracted Target Line")
    if result["target_line"]:
        st.success(result["target_line"])
        st.write(f"Confidence: **{result['confidence']:.3f}**")
    else:
        st.error("No `_1_` pattern detected in this image.")

    # Display highlighted image
    highlighted = draw_highlight(image, result["target_line"], result["all_lines"])
    st.subheader("🖼️ Highlighted Result")
    st.image(highlighted, channels="BGR", use_column_width=True)

    # Optionally show raw OCR output
    with st.expander("📄 Raw OCR Output"):
        st.json(result)