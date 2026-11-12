
import traceback

import cv2
import numpy as np

from .face_detection import get_face_detector
from .preprocessing import preprocess_face
from .emotion_recognition import get_emotion_recognizer
from .utils import get_emotion_color


def predict_emotions_in_frame(frame):
    """
    Detect faces, predict emotions, and return the annotated frame.
    """

    if frame is None:
        return {
            "faces_count": 0,
            "results": [],
            "annotated_frame": None,
            "model_loaded": False,
            "model_error": "No frame received.",
        }

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return {
            "faces_count": 0,
            "results": [],
            "annotated_frame": None,
            "model_loaded": False,
            "model_error": "Invalid or empty frame.",
        }

    print("\n--- Emotion analysis started ---")
    print("Input frame shape:", frame.shape)
    print("Input frame dtype:", frame.dtype)

    # Ensure a valid 3-channel image for annotation.
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Unsupported frame shape: {frame.shape}"
        )

    annotated_frame = frame.copy()

    # Detect faces.
    try:
        detector = get_face_detector()
        faces = detector.detect_faces(frame)
        print("Faces returned by detector:", len(faces))

    except Exception as exc:
        print("Face detection exception:", repr(exc))
        traceback.print_exc()
        raise

    results = []

    # Check recognizer state
    try:
        recognizer = get_emotion_recognizer()
    except Exception as exc:
        print("Recognizer initialization exception:", repr(exc))
        traceback.print_exc()
        raise

    if not recognizer.is_model_loaded:
        return {
            "faces_count": len(faces),
            "results": [],
            "annotated_frame": annotated_frame,
            "model_loaded": False,
            "model_error": recognizer.load_error or "Emotion recognition model is not loaded.",
        }

    # If no face is found, return a valid response.
    if not faces:
        return {
            "faces_count": 0,
            "results": [],
            "annotated_frame": annotated_frame,
            "model_loaded": True,
            "model_error": None,
        }

    for x, y, w, h in faces:
        try:
            # Keep bounding box inside frame
            x = max(0, int(x))
            y = max(0, int(y))
            w = min(int(w), frame.shape[1] - x)
            h = min(int(h), frame.shape[0] - y)

            if w <= 0 or h <= 0:
                continue

            # Apply 10% margin around face box to capture forehead and chin,
            # matching the distribution of FER-2013 training images, with safe image-boundary clamping.
            margin_x = int(round(w * 0.10))
            margin_y = int(round(h * 0.10))
            crop_x1 = max(0, x - margin_x)
            crop_y1 = max(0, y - margin_y)
            crop_x2 = min(frame.shape[1], x + w + margin_x)
            crop_y2 = min(frame.shape[0], y + h + margin_y)

            face_roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            if face_roi.size == 0:
                continue

            processed_roi = preprocess_face(face_roi)

            if processed_roi is None:
                print("Face preprocessing returned None.")
                continue

            emotion, confidence, all_scores = recognizer.predict(
                processed_roi
            )

            # Confidence tier categorization per requirements:
            # - 50% and above: Confident
            # - 35% to below 50%: Moderate Confidence
            # - Below 35%: Low Confidence / Ambiguous
            if confidence >= 50.0:
                confidence_tier = "Confident"
                is_uncertain = False
            elif confidence >= 35.0:
                confidence_tier = "Moderate"
                is_uncertain = False
            else:
                confidence_tier = "Low"
                is_uncertain = True

            display_emotion = str(emotion)

            color = get_emotion_color(emotion)

            # Adaptive line thickness and font sizing for resolution independence
            diag = float(np.sqrt(frame.shape[0] ** 2 + frame.shape[1] ** 2))
            line_thickness = max(2, int(round(diag / 450.0)))
            font_scale = max(0.5, min(1.2, float(w) / 250.0))
            font_thickness = max(1, int(round(font_scale * 2.0)))

            if confidence_tier == "Confident":
                label = f"{emotion}: {confidence:.1f}%"
            elif confidence_tier == "Moderate":
                label = f"{emotion}: {confidence:.1f}% (Mod)"
            else:
                label = f"{emotion}: {confidence:.1f}% (Low)"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            cv2.rectangle(
                annotated_frame,
                (x, y),
                (x + w, y + h),
                color,
                line_thickness,
            )

            # Position pill above face box, or below if near top of image
            pill_h = text_h + 10
            if y >= pill_h + 4:
                pill_y1 = max(0, y - pill_h - 2)
                pill_y2 = y
                text_y = y - 6
            else:
                pill_y1 = y + h
                pill_y2 = min(frame.shape[0] - 1, y + h + pill_h + 2)
                text_y = min(frame.shape[0] - 2, y + h + text_h + 4)

            pill_x2 = min(frame.shape[1] - 1, x + text_w + 12)

            cv2.rectangle(
                annotated_frame,
                (x, pill_y1),
                (pill_x2, pill_y2),
                color,
                -1,
            )

            cv2.putText(
                annotated_frame,
                label,
                (x + 6, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

            results.append({
                "emotion": str(emotion),
                "confidence": float(confidence),
                "confidence_tier": confidence_tier,
                "is_uncertain": is_uncertain,
                "display_emotion": display_emotion,
                "box": [x, y, w, h],
                "scores": all_scores,
            })

            print(
                f"[PREDICTION] Top Emotion: {emotion} | Confidence: {confidence:.2f}% | Tier: {confidence_tier} | LowConfidence: {is_uncertain}"
            )
            print(f"[SCORES] All 7 classes: {all_scores}")

        except Exception as exc:
            print("Face processing exception:", repr(exc))
            traceback.print_exc()
            raise

    print("--- Emotion analysis finished ---")

    return {
        "faces_count": len(faces),
        "results": results,
        "annotated_frame": annotated_frame,
        "model_loaded": bool(recognizer.is_model_loaded),
        "model_error": recognizer.load_error,
    }
