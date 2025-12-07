import cv2
import numpy as np

def load_image(path: str):
    """
    Load an image from disk.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def to_grayscale(img):
    """
    Convert BGR image to grayscale.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize_for_ocr(img, max_dim=1500):
    """
    Resize image so the longest side is 'max_dim' pixels.
    This helps OCR handle small/large images consistently.
    """
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:  # only shrink, don’t enlarge
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return img

def denoise(gray):
    """
    Light Gaussian blur to reduce noise.
    """
    return cv2.GaussianBlur(gray, (3, 3), 0)

def threshold(gray):
    """
    Adaptive threshold for better text segmentation.
    """
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

def preprocess_image(path: str):
    """
    Full preprocessing pipeline.
    Returns: (original_image, processed_image)
    """
    img = load_image(path)
    img = resize_for_ocr(img)
    gray = to_grayscale(img)
    denoised = denoise(gray)
    th = threshold(denoised)
    return img, th


def preprocess_image_array(img):
    """
    Preprocess an already loaded image (NumPy array).
    Returns (orig, processed)
    """
    img_resized = resize_for_ocr(img)
    gray = to_grayscale(img_resized)
    denoised = denoise(gray)
    th = threshold(denoised)
    return img_resized, th