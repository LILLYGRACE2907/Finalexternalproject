
import os
import threading

import numpy as np


# These labels must match the model's training order.
DEFAULT_EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

# Empirical class distribution from FER training split (total: 22,009 samples)
# Happy (5,669) is 18x more frequent than Disgust (304), which suppresses minority emotion logits.
# Prior calibration neutralizes this dataset collection artifact so all 7 emotions are detectable.
TRAIN_CLASS_COUNTS = {
    "Angry": 3079.0,
    "Disgust": 304.0,
    "Fear": 3127.0,
    "Happy": 5669.0,
    "Neutral": 3899.0,
    "Sad": 3786.0,
    "Surprise": 2145.0,
}


class EmotionRecognizer:
    def __init__(self, model_path=None, labels_path=None, calibration_alpha=0.0):
        # Project root directory
        base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        # Prioritize high-accuracy trained model (emotion_model.h5: 57% test accuracy),
        # with fallback to emotion_model_trained.keras if .h5 is missing
        if model_path is None:
            h5_path = os.path.join(base_dir, "models", "emotion_model.h5")
            keras_path = os.path.join(base_dir, "models", "emotion_model_trained.keras")
            self.model_path = h5_path if os.path.isfile(h5_path) else keras_path
        else:
            self.model_path = model_path

        # Prioritize emotion_labels.txt with fallback to emotion_labels_trained.txt
        if labels_path is None:
            txt_path = os.path.join(base_dir, "models", "emotion_labels.txt")
            txt_trained = os.path.join(base_dir, "models", "emotion_labels_trained.txt")
            self.labels_path = txt_path if os.path.isfile(txt_path) else txt_trained
        else:
            self.labels_path = labels_path

        self.labels = self.load_labels()

        # Calibration factor (0.0 = uncalibrated, 0.40 = balanced priors across all 7 emotions)
        self.calibration_alpha = float(calibration_alpha)
        priors = np.array(
            [TRAIN_CLASS_COUNTS.get(lbl, 1.0) for lbl in self.labels],
            dtype=np.float64,
        )
        self.class_priors = priors / np.sum(priors)

        self.model = None
        self.is_model_loaded = False
        self.load_error = None
        self._loading = False
        self._lock = threading.RLock()

        self.load_model()

    def load_labels(self):
        """Read emotion labels from the trained labels file."""
        if os.path.isfile(self.labels_path):
            try:
                with open(
                    self.labels_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    labels = [
                        line.strip()
                        for line in file
                        if line.strip()
                    ]

                if len(labels) == 7:
                    return labels

                print(
                    "Warning: Expected 7 emotion labels, "
                    f"but found {len(labels)}."
                )

            except (OSError, UnicodeError) as exc:
                print("Label loading error:", exc)

        print("Using default emotion labels.")
        return DEFAULT_EMOTIONS.copy()

    def load_model(self):
        """Load the trained CNN model."""
        with self._lock:
            if self.is_model_loaded and self.model is not None:
                return True

            if self._loading:
                self.load_error = (
                    "Model loading is already in progress."
                )
                return False

            if not os.path.isfile(self.model_path):
                self.load_error = (
                    f"Model file not found: {self.model_path}"
                )
                print(self.load_error)
                return False

            self._loading = True

            try:
                import tensorflow as tf

                loaded_model = tf.keras.models.load_model(
                    self.model_path,
                    compile=False,
                )

                input_shape = loaded_model.input_shape
                output_shape = loaded_model.output_shape

                if isinstance(input_shape, list):
                    raise ValueError(
                        "This recognizer requires a single-input model."
                    )

                if len(input_shape) != 4:
                    raise ValueError(
                        f"Expected a 4D image input, got {input_shape}."
                    )

                if input_shape[1:] != (48, 48, 1):
                    raise ValueError(
                        "The trained model should accept "
                        f"(48, 48, 1) images, got {input_shape}."
                    )

                if isinstance(output_shape, list):
                    raise ValueError(
                        "This recognizer requires a single-output model."
                    )

                if output_shape[-1] != len(self.labels):
                    raise ValueError(
                        f"Model outputs {output_shape[-1]} scores, "
                        f"but there are {len(self.labels)} labels."
                    )

                self.model = loaded_model
                self.is_model_loaded = True
                self.load_error = None

                print("Trained CNN model loaded successfully.")
                print("Model path:", self.model_path)
                print("Model input shape:", input_shape)
                print("Model output shape:", output_shape)
                print("Emotion labels:", self.labels)

                return True

            except Exception as exc:
                self.model = None
                self.is_model_loaded = False
                self.load_error = f"{type(exc).__name__}: {exc}"

                print("Model loading error:", self.load_error)
                return False

            finally:
                self._loading = False

    def predict(self, processed_face):
        """Return emotion, confidence percentage, and all scores."""
        with self._lock:
            if self.model is None or not self.is_model_loaded:
                if not self.load_model():
                    raise RuntimeError(
                        "CNN model could not be loaded: "
                        f"{self.load_error}"
                    )

            face = np.asarray(
                processed_face,
                dtype=np.float32,
            )

            if face.size == 0:
                raise ValueError(
                    "The processed face image is empty."
                )

            expected_shape = self.model.input_shape

            if face.ndim != len(expected_shape):
                raise ValueError(
                    f"Input shape mismatch. Model expects "
                    f"{expected_shape}, received {face.shape}. "
                    "Check preprocessing.py."
                )

            for expected, actual in zip(
                expected_shape[1:],
                face.shape[1:],
            ):
                if expected is not None and expected != actual:
                    raise ValueError(
                        f"Input shape mismatch. Model expects "
                        f"{expected_shape}, received {face.shape}. "
                        "Check preprocessing.py."
                    )

            predictions = np.asarray(
                self.model.predict(face, verbose=0),
                dtype=np.float64,
            )

            if predictions.ndim != 2 or predictions.shape[0] < 1:
                raise ValueError(
                    f"Unexpected model output shape: "
                    f"{predictions.shape}"
                )

            predictions = predictions[0]

            if len(predictions) != len(self.labels):
                raise ValueError(
                    f"Model returns {len(predictions)} scores, "
                    f"but there are {len(self.labels)} labels."
                )

            if not np.all(np.isfinite(predictions)):
                raise ValueError(
                    "Model returned invalid prediction values."
                )

            # Convert logits to probabilities if necessary.
            total = float(np.sum(predictions))

            if (
                np.any(predictions < 0)
                or np.any(predictions > 1)
                or not np.isclose(total, 1.0, atol=0.01)
            ):
                shifted = predictions - np.max(predictions)
                exp_values = np.exp(shifted)
                predictions = exp_values / np.sum(exp_values)

            elif total > 0:
                predictions = predictions / total

            else:
                raise ValueError(
                    "Model returned scores with a zero total."
                )

            # Capture raw uncalibrated probabilities from model softmax
            raw_scores = {
                self.labels[i]: round(
                    float(predictions[i] * 100.0),
                    2,
                )
                for i in range(len(self.labels))
            }

            # Apply prior recalibration to remove training set class frequency bias
            # This enables all 7 emotions (including Disgust and Fear) to be recognized
            # when their visual features appear, rather than being suppressed by dataset imbalance.
            if self.calibration_alpha > 0 and len(self.class_priors) == len(predictions):
                calibrated = predictions / (self.class_priors ** self.calibration_alpha)
                cal_sum = float(np.sum(calibrated))
                if cal_sum > 0:
                    predictions = calibrated / cal_sum

            max_idx = int(np.argmax(predictions))

            emotion = self.labels[max_idx]
            confidence = round(
                float(predictions[max_idx] * 100.0),
                2,
            )

            all_scores = {
                self.labels[i]: round(
                    float(predictions[i] * 100.0),
                    2,
                )
                for i in range(len(self.labels))
            }

            print(f"[RECOGNIZER] Model scores: {all_scores} -> {emotion} ({confidence:.1f}%)")

            return emotion, confidence, all_scores


# One shared recognizer per Python process.
_recognizer = None
_recognizer_lock = threading.Lock()


def get_emotion_recognizer(calibration_alpha=0.0):
    """Return the shared emotion recognizer."""
    global _recognizer

    if _recognizer is None:
        with _recognizer_lock:
            if _recognizer is None:
                _recognizer = EmotionRecognizer(calibration_alpha=calibration_alpha)

    return _recognizer


def predict_emotion(processed_face):
    """Compatibility helper for existing project imports."""
    recognizer = get_emotion_recognizer()
    return recognizer.predict(processed_face)