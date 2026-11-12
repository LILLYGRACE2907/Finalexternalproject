import os
import threading
import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        self.face_cascade = None      # Default cascade (preserves compatibility)
        self.alt_cascade = None       # haarcascade_frontalface_alt.xml
        self.alt2_cascade = None      # haarcascade_frontalface_alt2.xml
        self.profile_cascade = None   # haarcascade_profileface.xml
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.is_available = False
        self._lock = threading.Lock()
        self.load_detector()

    def load_detector(self):
        """Load OpenCV Haar Cascade face detectors with multi-model fallback."""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local_xml = os.path.join(base_dir, "models", "face_detector.xml")
            cv_dir = cv2.data.haarcascades
            cv_default = os.path.join(cv_dir, "haarcascade_frontalface_default.xml")
            cv_alt = os.path.join(cv_dir, "haarcascade_frontalface_alt.xml")
            cv_alt2 = os.path.join(cv_dir, "haarcascade_frontalface_alt2.xml")
            cv_profile = os.path.join(cv_dir, "haarcascade_profileface.xml")

            # 1. Load default cascade (local model or OpenCV default)
            default_cascade = None
            if os.path.isfile(local_xml):
                print("Loading Haar Cascade from repo:", local_xml)
                try:
                    loaded = cv2.CascadeClassifier(local_xml)
                    if not loaded.empty():
                        default_cascade = loaded
                except Exception as xml_err:
                    print("Local face_detector.xml load error:", xml_err)

            if default_cascade is None and os.path.isfile(cv_default):
                print("Loading Haar Cascade from OpenCV:", cv_default)
                loaded = cv2.CascadeClassifier(cv_default)
                if not loaded.empty():
                    default_cascade = loaded

            self.face_cascade = default_cascade

            # 2. Load frontalface_alt (classic Gentle AdaBoost, exceptional on frontal faces)
            if os.path.isfile(cv_alt):
                try:
                    loaded_alt = cv2.CascadeClassifier(cv_alt)
                    if not loaded_alt.empty():
                        self.alt_cascade = loaded_alt
                except Exception as alt_err:
                    print("Alt cascade load notice:", alt_err)

            # 3. Load frontalface_alt2 (Gentle AdaBoost on trees, robust across lighting)
            if os.path.isfile(cv_alt2):
                try:
                    loaded_alt2 = cv2.CascadeClassifier(cv_alt2)
                    if not loaded_alt2.empty():
                        self.alt2_cascade = loaded_alt2
                except Exception as alt2_err:
                    print("Alt2 cascade load notice:", alt2_err)

            # 4. Load profileface (detects angled or turned faces)
            if os.path.isfile(cv_profile):
                try:
                    loaded_profile = cv2.CascadeClassifier(cv_profile)
                    if not loaded_profile.empty():
                        self.profile_cascade = loaded_profile
                except Exception as prof_err:
                    print("Profile cascade load notice:", prof_err)

            # Verify at least one valid cascade loaded
            if (self.face_cascade is None or self.face_cascade.empty()) and \
               (self.alt_cascade is None or self.alt_cascade.empty()) and \
               (self.alt2_cascade is None or self.alt2_cascade.empty()):
                print("ERROR: Haar Cascade failed to load from any source.")
                self.is_available = False
                return False

            self.is_available = True
            print("SUCCESS: Haar Cascade loaded.")
            return True

        except Exception as exc:
            print("Cascade loading error:", repr(exc))
            self.face_cascade = None
            self.is_available = False
            return False

    @staticmethod
    def apply_nms(boxes, iou_threshold=0.35):
        """
        Non-Maximum Suppression (NMS) using Intersection-over-Union (IoU).
        Eliminates redundant, duplicate, and nested bounding boxes while preserving distinct faces.
        """
        if len(boxes) <= 1:
            return boxes

        # Sort boxes by area descending
        sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        keep = []

        while sorted_boxes:
            current = sorted_boxes.pop(0)
            keep.append(current)
            x1, y1, w1, h1 = current
            area1 = float(w1 * h1)

            filtered = []
            for b in sorted_boxes:
                x2, y2, w2, h2 = b
                area2 = float(w2 * h2)

                # Intersection rectangle
                ix1 = max(x1, x2)
                iy1 = max(y1, y2)
                ix2 = min(x1 + w1, x2 + w2)
                iy2 = min(y1 + h1, y2 + h2)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection = float(iw * ih)

                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0.0
                min_area = min(area1, area2)
                containment = intersection / min_area if min_area > 0 else 0.0

                # Suppress if boxes heavily overlap or one is largely contained inside the other
                if iou < iou_threshold and containment < 0.60:
                    filtered.append(b)

            sorted_boxes = filtered

        return keep

    def detect_faces(self, frame):
        """
        Return detected faces as (x, y, width, height) tuples sorted by area descending.
        Employs a robust multi-stage cascade with adaptive contrast (CLAHE), fine scaleFactor,
        and coordinate preservation.
        """
        if frame is None:
            print("Face detection skipped: frame is None.")
            return []

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            print("Face detection skipped: invalid or empty frame.")
            return []

        if frame.ndim not in (2, 3):
            print("Face detection skipped: unsupported frame dimensions.")
            return []

        try:
            orig_h, orig_w = frame.shape[:2]

            # Convert to grayscale
            if frame.ndim == 2:
                gray = frame
            elif frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            else:
                print("Face detection skipped: unsupported channels.")
                return []

            if gray.dtype != np.uint8:
                gray = np.nan_to_num(gray, nan=0, posinf=255, neginf=0)
                gray = np.clip(gray, 0, 255).astype(np.uint8)

            gray = np.ascontiguousarray(gray).copy()

            if orig_h < 20 or orig_w < 20:
                print("Face detection skipped: frame is too small.")
                return []

            if not self.is_available:
                print("Face detector unavailable; retrying initialization.")
                if not self.load_detector():
                    return []

            # Resolution scaling for large images to optimize speed while preserving coordinates
            scale_factor_img = 1.0
            max_dim = max(orig_h, orig_w)
            if max_dim > 1600:
                scale_factor_img = 1600.0 / float(max_dim)
                proc_w = max(30, int(round(orig_w * scale_factor_img)))
                proc_h = max(30, int(round(orig_h * scale_factor_img)))
                detect_gray = cv2.resize(gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            else:
                detect_gray = gray

            min_face_size = (max(24, int(36 * scale_factor_img)), max(24, int(36 * scale_factor_img)))

            detected = []
            detection_source = None

            # Thread-safe multi-stage detection
            with self._lock:
                # Stage 1: Frontalface Alt with robust scaleFactor 1.1 and minNeighbors 5
                # Reliable on frontal faces while preventing false detections on clutter
                if self.alt_cascade is not None and not self.alt_cascade.empty():
                    detected = self.alt_cascade.detectMultiScale(
                        detect_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=min_face_size,
                        flags=cv2.CASCADE_SCALE_IMAGE,
                    )
                    if len(detected) > 0:
                        detection_source = "frontalface_alt (sf=1.1, mn=5)"

                # Stage 2: Frontalface Alt2 with scaleFactor 1.1 and minNeighbors 5
                if len(detected) == 0 and self.alt2_cascade is not None and not self.alt2_cascade.empty():
                    detected = self.alt2_cascade.detectMultiScale(
                        detect_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=min_face_size,
                        flags=cv2.CASCADE_SCALE_IMAGE,
                    )
                    if len(detected) > 0:
                        detection_source = "frontalface_alt2 (sf=1.1, mn=5)"

                # Stage 3: Frontalface Default with scaleFactor 1.1 and minNeighbors 5
                if len(detected) == 0 and self.face_cascade is not None and not self.face_cascade.empty():
                    detected = self.face_cascade.detectMultiScale(
                        detect_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=min_face_size,
                        flags=cv2.CASCADE_SCALE_IMAGE,
                    )
                    if len(detected) > 0:
                        detection_source = "frontalface_default (sf=1.1, mn=5)"

                # Stage 4: CLAHE (Contrast-Limited Adaptive Histogram Equalization)
                # Recovers faces in low-contrast, shadowed, or backlit conditions
                if len(detected) == 0:
                    try:
                        clahe_gray = self.clahe.apply(detect_gray)
                        if self.alt_cascade is not None:
                            detected = self.alt_cascade.detectMultiScale(
                                clahe_gray,
                                scaleFactor=1.1,
                                minNeighbors=5,
                                minSize=min_face_size,
                                flags=cv2.CASCADE_SCALE_IMAGE,
                            )
                            if len(detected) > 0:
                                detection_source = "CLAHE + frontalface_alt (sf=1.1, mn=5)"

                        if len(detected) == 0 and self.alt2_cascade is not None:
                            detected = self.alt2_cascade.detectMultiScale(
                                clahe_gray,
                                scaleFactor=1.1,
                                minNeighbors=5,
                                minSize=min_face_size,
                                flags=cv2.CASCADE_SCALE_IMAGE,
                            )
                            if len(detected) > 0:
                                detection_source = "CLAHE + frontalface_alt2 (sf=1.1, mn=5)"

                        if len(detected) == 0 and self.face_cascade is not None:
                            detected = self.face_cascade.detectMultiScale(
                                clahe_gray,
                                scaleFactor=1.1,
                                minNeighbors=5,
                                minSize=min_face_size,
                                flags=cv2.CASCADE_SCALE_IMAGE,
                            )
                            if len(detected) > 0:
                                detection_source = "CLAHE + frontalface_default (sf=1.1, mn=5)"
                    except Exception as clahe_err:
                        print("CLAHE stage notice:", clahe_err)

                # Stage 5: Global Histogram Equalization fallback
                if len(detected) == 0:
                    try:
                        eq_gray = cv2.equalizeHist(detect_gray)
                        if self.alt_cascade is not None:
                            detected = self.alt_cascade.detectMultiScale(
                                eq_gray,
                                scaleFactor=1.1,
                                minNeighbors=4,
                                minSize=min_face_size,
                                flags=cv2.CASCADE_SCALE_IMAGE,
                            )
                            if len(detected) > 0:
                                detection_source = "EqualizeHist + frontalface_alt (sf=1.1, mn=4)"

                        if len(detected) == 0 and self.face_cascade is not None:
                            detected = self.face_cascade.detectMultiScale(
                                eq_gray,
                                scaleFactor=1.1,
                                minNeighbors=4,
                                minSize=min_face_size,
                                flags=cv2.CASCADE_SCALE_IMAGE,
                            )
                            if len(detected) > 0:
                                detection_source = "EqualizeHist + frontalface_default (sf=1.1, mn=4)"
                    except Exception as eq_err:
                        print("Histogram equalization notice:", eq_err)

                # Stage 6: Profile Face Detection (angled or side-view faces)
                if len(detected) == 0 and self.profile_cascade is not None and not self.profile_cascade.empty():
                    try:
                        prof_detected = self.profile_cascade.detectMultiScale(
                            detect_gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=min_face_size,
                            flags=cv2.CASCADE_SCALE_IMAGE,
                        )
                        if len(prof_detected) > 0:
                            detected = prof_detected
                            detection_source = "profileface (sf=1.1, mn=5)"
                        else:
                            # Test horizontally flipped for opposite profile
                            flipped_gray = cv2.flip(detect_gray, 1)
                            flip_detected = self.profile_cascade.detectMultiScale(
                                flipped_gray,
                                scaleFactor=1.1,
                                minNeighbors=5,
                                minSize=min_face_size,
                                flags=cv2.CASCADE_SCALE_IMAGE,
                            )
                            if len(flip_detected) > 0:
                                pw = detect_gray.shape[1]
                                detected = [[pw - (bx + bw), by, bw, bh] for (bx, by, bw, bh) in flip_detected]
                                detection_source = "profileface_flipped (sf=1.1, mn=5)"
                    except Exception as prof_err:
                        print("Profile cascade notice:", prof_err)

            if detection_source:
                print(f"[DETECTION] Stage: {detection_source} -> Raw count: {len(detected)}")

            # Coordinate mapping and validation
            faces = []
            for item in detected:
                x, y, w, h = item
                if scale_factor_img != 1.0:
                    x = int(round(x / scale_factor_img))
                    y = int(round(y / scale_factor_img))
                    w = int(round(w / scale_factor_img))
                    h = int(round(h / scale_factor_img))

                # Clamp coordinates strictly inside original frame
                x = max(0, min(x, orig_w - 1))
                y = max(0, min(y, orig_h - 1))
                w = max(1, min(w, orig_w - x))
                h = max(1, min(h, orig_h - y))

                # Discard extreme aspect ratios (faces are roughly square)
                ratio = float(w) / float(h)
                if ratio < 0.55 or ratio > 1.8:
                    continue

                faces.append((int(x), int(y), int(w), int(h)))

            # Apply IoU Non-Maximum Suppression (NMS) to eliminate duplicate/nested bounding boxes
            faces = self.apply_nms(faces, iou_threshold=0.35)

            # Sort detected faces by area descending so dominant face is always at index 0
            faces.sort(key=lambda b: b[2] * b[3], reverse=True)

            print("Faces found:", len(faces))
            return faces

        except cv2.error as exc:
            print("OpenCV face detection error:", repr(exc))
            return []

        except Exception as exc:
            print("Unexpected face detection error:", repr(exc))
            return []


_face_detector = None
_detector_lock = threading.Lock()


def get_face_detector():
    """Return a shared FaceDetector instance."""
    global _face_detector

    if _face_detector is None:
        with _detector_lock:
            if _face_detector is None:
                _face_detector = FaceDetector()

    return _face_detector


def detect_faces(frame):
    """Compatibility function for existing project imports."""
    return get_face_detector().detect_faces(frame)
