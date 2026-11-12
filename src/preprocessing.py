import cv2
import numpy as np

def preprocess_face(face_roi, target_size=(48, 48)):
    """
    Preprocess face region of interest for emotion recognition model.
    Steps:
    1. Convert to grayscale if image is color.
    2. Resize image to target size (e.g. 48x48).
    3. Normalize pixel values to [0, 1].
    4. Reshape to format suitable for Keras model input: (1, height, width, 1).
    """
    if face_roi is None or face_roi.size == 0:
        return None

    # Convert to grayscale if multi-channel
    if len(face_roi.shape) == 3:
        if face_roi.shape[2] == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        elif face_roi.shape[2] == 4:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGRA2GRAY)
        elif face_roi.shape[2] == 1:
            gray = face_roi[:, :, 0]
        else:
            gray = face_roi[:, :, 0]
    else:
        gray = face_roi

    # Ensure square aspect ratio without stretching/squashing facial features
    h, w = gray.shape[:2]
    if h != w:
        diff = abs(h - w)
        if h > w:
            pad_left = diff // 2
            pad_right = diff - pad_left
            gray = cv2.copyMakeBorder(gray, 0, 0, pad_left, pad_right, cv2.BORDER_REPLICATE)
        else:
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            gray = cv2.copyMakeBorder(gray, pad_top, pad_bottom, 0, 0, cv2.BORDER_REPLICATE)

    # Resize to target size (48x48) with interpolation suited for scaling direction
    if gray.shape[0] >= target_size[0] and gray.shape[1] >= target_size[1]:
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize pixels to range [0, 1] as required by training
    normalized = resized.astype('float32') / 255.0

    # Expand dimensions for model prediction (batch_size=1, height=48, width=48, channels=1)
    expanded = np.expand_dims(normalized, axis=(0, -1))

    return expanded
