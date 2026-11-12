import os
import cv2
import numpy as np

# Emotion Color Mapping (BGR format for OpenCV, Hex for CSS)
EMOTION_COLORS_BGR = {
    'Angry': (0, 0, 255),       # Red
    'Disgust': (0, 140, 255),    # Orange
    'Fear': (128, 0, 128),      # Purple
    'Happy': (0, 255, 0),       # Green
    'Sad': (255, 100, 0),       # Blue
    'Surprise': (255, 255, 0),  # Cyan
    'Neutral': (200, 200, 200)  # Light Gray
}

EMOTION_COLORS_HEX = {
    'Angry': '#dc3545',
    'Disgust': '#fd7e14',
    'Fear': '#6f42c1',
    'Happy': '#198754',
    'Sad': '#0d6efd',
    'Surprise': '#0dcaf0',
    'Neutral': '#6c757d'
}

def get_emotion_color(emotion):
    return EMOTION_COLORS_BGR.get(emotion, (0, 255, 0))

def get_emotion_hex(emotion):
    return EMOTION_COLORS_HEX.get(emotion, '#6c757d')

def check_model_files():
    """
    Check if required AI model files exist in the models/ directory.
    Supports emotion_model.h5 (primary) or emotion_model_trained.keras,
    and emotion_labels.txt (primary) or emotion_labels_trained.txt.
    Returns (status_bool, missing_files_list)
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')

    xml_ok = (
        os.path.isfile(os.path.join(models_dir, 'face_detector.xml'))
        or os.path.isfile(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
    )
    model_ok = (
        os.path.isfile(os.path.join(models_dir, 'emotion_model.h5'))
        or os.path.isfile(os.path.join(models_dir, 'emotion_model_trained.keras'))
    )
    labels_ok = (
        os.path.isfile(os.path.join(models_dir, 'emotion_labels.txt'))
        or os.path.isfile(os.path.join(models_dir, 'emotion_labels_trained.txt'))
    )

    missing = []
    if not xml_ok:
        missing.append('face_detector.xml')
    if not model_ok:
        missing.append('emotion_model.h5 / emotion_model_trained.keras')
    if not labels_ok:
        missing.append('emotion_labels.txt / emotion_labels_trained.txt')

    return (len(missing) == 0), missing

def ensure_default_models():
    """
    Ensure models/ folder has face_detector.xml and emotion_labels.txt.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 1. Face detector XML
    xml_target = os.path.join(models_dir, 'face_detector.xml')
    if not os.path.exists(xml_target):
        default_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(default_xml):
            with open(default_xml, 'rb') as f_in, open(xml_target, 'wb') as f_out:
                f_out.write(f_in.read())

    # 2. Emotion labels text
    labels_target = os.path.join(models_dir, 'emotion_labels.txt')
    if not os.path.exists(labels_target):
        labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        with open(labels_target, 'w') as f:
            f.write('\n'.join(labels))
