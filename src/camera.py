import cv2
import time
import numpy as np
from .prediction import predict_emotions_in_frame


class VideoCamera:

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.video = None
        self.is_available = False

        self.open_camera()

    def open_camera(self):
        try:
            try:
                self.video = cv2.VideoCapture(
                    self.camera_index,
                    cv2.CAP_DSHOW
                )
            except Exception as dshow_err:
                print(f"DSHOW backend failed for camera {self.camera_index}: {dshow_err}")
                self.video = None

            if self.video is None or not self.video.isOpened():
                self.video = cv2.VideoCapture(self.camera_index)

            if self.video.isOpened():
                # Prefer 1280x720 HD for modern widescreen laptop webcams
                self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                actual_w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera opened successfully. Resolution: {actual_w}x{actual_h}")

                success, frame = self.video.read()

                if success and frame is not None:
                    self.is_available = True
                else:
                    self.is_available = False
                    print("Camera opened but frame capture failed")

            else:
                self.is_available = False
                print("Camera could not be opened")

        except Exception as e:
            self.is_available = False
            print("Camera error:", e)

    def get_frame(self):

        # -----------------------------------------
        # CHECK CAMERA
        # -----------------------------------------
        if (
            not self.is_available
            or self.video is None
            or not self.video.isOpened()
        ):

            placeholder = self.create_placeholder_frame(
                "Camera Unavailable"
            )

            ret, jpeg = cv2.imencode(
                '.jpg',
                placeholder
            )

            return jpeg.tobytes(), {
                'faces_count': 0,
                'results': [],
                'model_loaded': False
            }

        # -----------------------------------------
        # CAPTURE FRAME
        # -----------------------------------------
        success, frame = self.video.read()

        if not success or frame is None:

            placeholder = self.create_placeholder_frame(
                "Failed to capture frame"
            )

            ret, jpeg = cv2.imencode(
                '.jpg',
                placeholder
            )

            return jpeg.tobytes(), {
                'faces_count': 0,
                'results': [],
                'model_loaded': False
            }

        # -----------------------------------------
        # EMOTION PREDICTION
        # -----------------------------------------
        try:

            prediction_info = predict_emotions_in_frame(frame)

            # Get frame with face box + emotion label
            annotated_frame = prediction_info.get(
                'annotated_frame',
                frame
            )

        except Exception as e:

            print("Prediction error:", e)

            prediction_info = {
                'faces_count': 0,
                'results': [],
                'model_loaded': False,
                'error': str(e)
            }

            # If prediction fails,
            # still show normal camera
            annotated_frame = frame

        # -----------------------------------------
        # ENCODE FRAME
        # -----------------------------------------
        ret, jpeg = cv2.imencode(
            '.jpg',
            annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 92]
        )

        if not ret:
            return b'', prediction_info

        return jpeg.tobytes(), prediction_info

    def create_placeholder_frame(
        self,
        message="No Camera"
    ):

        # -----------------------------------------
        # BACKGROUND
        # -----------------------------------------
        img = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

        img[:] = (30, 25, 20)

        # -----------------------------------------
        # WARNING BOX
        # -----------------------------------------
        cv2.rectangle(
            img,
            (50, 150),
            (590, 330),
            (50, 50, 60),
            -1
        )

        cv2.rectangle(
            img,
            (50, 150),
            (590, 330),
            (0, 165, 255),
            2
        )

        # -----------------------------------------
        # TITLE
        # -----------------------------------------
        cv2.putText(
            img,
            "FACIAL EMOTION RECOGNITION SYSTEM",
            (70, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # -----------------------------------------
        # STATUS
        # -----------------------------------------
        cv2.putText(
            img,
            f"Status: {message}",
            (70, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 215, 255),
            2
        )

        # -----------------------------------------
        # MESSAGE
        # -----------------------------------------
        cv2.putText(
            img,
            "Please connect a webcam",
            (70, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )

        return img

    def release(self):

        if (
            self.video is not None
            and self.video.isOpened()
        ):
            self.video.release()

        self.is_available = False

        print("Camera released")


def gen_frames(camera):

    """
    Generator function for Django
    StreamingHttpResponse.
    """

    while True:

        try:

            frame_bytes, info = camera.get_frame()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + frame_bytes
                + b'\r\n\r\n'
            )

            # Approximately 25 FPS
            time.sleep(0.04)

        except Exception as e:

            print("Streaming error:", e)
            break