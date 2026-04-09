import cv2
import numpy as np
try:
    from tensorflow.keras.applications.inception_v3 import preprocess_input
except ImportError:
    from keras.applications.inception_v3 import preprocess_input

TARGET_SIZE = (299, 299)

def localise_and_crop(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_gray

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    H, W = img_gray.shape

    pad_top    = int(0.15 * h)
    pad_bottom = int(0.10 * h)
    pad_left   = int(0.08 * w)
    pad_right  = int(0.08 * w)

    x1 = max(x - pad_left,   0)
    y1 = max(y - pad_top,    0)
    x2 = min(x + w + pad_right,  W)
    y2 = min(y + h + pad_bottom, H)

    crop = img_gray[y1:y2, x1:x2]
    return crop if crop.size > 0 else img_gray


def apply_clahe(img_gray, clip_limit=3, tile_grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img_gray)


def apply_gaussian_blur(img_gray, kernel_size=(3, 3), sigma=0):
    return cv2.GaussianBlur(img_gray, kernel_size, sigma)


def preprocess_image_from_bytes(img_bytes, target_size=TARGET_SIZE):
    """
    Accepts raw image bytes (from st.file_uploader),
    runs the full preprocessing pipeline, and returns
    a float32 array of shape (1, H, W, 3) ready for model.predict().
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return np.zeros((1, *target_size, 3), dtype=np.float32)

    img = localise_and_crop(img)
    img = apply_clahe(img)
    img = apply_gaussian_blur(img)
    img = cv2.resize(img, target_size)
    img = cv2.merge([img, img, img])
    img = img.astype(np.float32)
    img = preprocess_input(img)          # InceptionV3 normalization

    return np.expand_dims(img, axis=0)  # Shape: (1, 299, 299, 3)