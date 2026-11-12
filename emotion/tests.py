from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from .models import EmotionRecognition, ContactMessage

class EmotionSystemTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser@example.com',
            email='testuser@example.com',
            password='testpassword123',
            first_name='Test',
            last_name='User'
        )

    def test_home_page(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'FACIAL EMOTION RECOGNITION SYSTEM')

    def test_about_page(self):
        response = self.client.get(reverse('about'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'About Facial Emotion Recognition')

    def test_contact_page_get_and_post(self):
        response = self.client.get(reverse('contact'))
        self.assertEqual(response.status_code, 200)

        post_data = {
            'name': 'Tester',
            'email': 'tester@example.com',
            'message': 'Testing contact submission'
        }
        post_response = self.client.post(reverse('contact'), post_data)
        self.assertEqual(post_response.status_code, 302)
        self.assertTrue(ContactMessage.objects.filter(email='tester@example.com').exists())

    def test_registration_and_login(self):
        # Registration
        reg_data = {
            'full_name': 'New Student',
            'email': 'newstudent@example.com',
            'password': 'newpassword123',
            'confirm_password': 'newpassword123'
        }
        reg_response = self.client.post(reverse('register'), reg_data)
        self.assertEqual(reg_response.status_code, 302)
        self.assertTrue(User.objects.filter(email='newstudent@example.com').exists())

        # Login
        login_data = {
            'email': 'newstudent@example.com',
            'password': 'newpassword123'
        }
        login_response = self.client.post(reverse('login'), login_data)
        self.assertEqual(login_response.status_code, 302)

    def test_protected_views_require_login(self):
        # Unauthenticated access should redirect to login
        for view_name in ['dashboard', 'recognition', 'history']:
            response = self.client.get(reverse(view_name))
            self.assertEqual(response.status_code, 302)

        # Authenticated access
        self.client.login(username='testuser@example.com', password='testpassword123')
        for view_name in ['dashboard', 'recognition', 'history']:
            response = self.client.get(reverse(view_name))
            self.assertEqual(response.status_code, 200)

    def test_save_recognition_api(self):
        self.client.login(username='testuser@example.com', password='testpassword123')
        save_data = {
            'emotion': 'Happy',
            'confidence': 98.5
        }
        response = self.client.post(
            reverse('save_recognition'),
            data=save_data,
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(EmotionRecognition.objects.filter(user=self.user, emotion='Happy').exists())

    def test_recognition_page_elements(self):
        self.client.login(username='testuser@example.com', password='testpassword123')
        response = self.client.get(reverse('recognition'))
        self.assertEqual(response.status_code, 200)
        # Check required webcam and UI elements
        self.assertContains(response, 'id="webcamVideo"')
        self.assertContains(response, 'id="overlayCanvas"')
        self.assertContains(response, 'id="btnStartCam"')
        self.assertContains(response, 'id="btnStopCam"')
        self.assertContains(response, 'id="btnCapture"')
        self.assertContains(response, 'id="btnSaveResult"')
        self.assertContains(response, 'id="fileInput"')
        self.assertContains(response, 'navigator.mediaDevices.getUserMedia')
        # Check all 7 emotion classes are present
        for emo in ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']:
            self.assertContains(response, emo)

    def test_process_frame_api_with_valid_image(self):
        import base64
        import cv2
        import numpy as np

        self.client.login(username='testuser@example.com', password='testpassword123')

        # Create a synthetic image for testing
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', test_img)
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        response = self.client.post(
            reverse('process_frame'),
            data={'image': img_b64},
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertTrue(json_data['success'])
        self.assertIn('faces_count', json_data)
        self.assertIn('results', json_data)
        self.assertIn('model_loaded', json_data)

    def test_process_frame_api_validation(self):
        self.client.login(username='testuser@example.com', password='testpassword123')

        # GET request should return 405 Method Not Allowed
        get_response = self.client.get(reverse('process_frame'))
        self.assertEqual(get_response.status_code, 405)

        # Empty body should return 400 Bad Request
        empty_response = self.client.post(
            reverse('process_frame'),
            data={'image': ''},
            content_type='application/json'
        )
        self.assertEqual(empty_response.status_code, 400)

    def test_process_frame_multipart_file_upload(self):
        import cv2
        import numpy as np
        from django.core.files.uploadedfile import SimpleUploadedFile

        self.client.login(username='testuser@example.com', password='testpassword123')

        # Create a valid JPEG file in memory
        test_img = np.zeros((120, 120, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', test_img)
        uploaded_file = SimpleUploadedFile("sample.jpg", buffer.tobytes(), content_type="image/jpeg")

        response = self.client.post(
            reverse('process_frame'),
            {'image': uploaded_file}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('faces_count', data)
        self.assertIn('annotated_image', data)

    def test_process_frame_invalid_file_extension(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        self.client.login(username='testuser@example.com', password='testpassword123')

        invalid_file = SimpleUploadedFile("document.pdf", b"%PDF-1.4 test data", content_type="application/pdf")
        response = self.client.post(
            reverse('process_frame'),
            {'image': invalid_file}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data['success'])
        self.assertIn('Unsupported file format', data['error'])

    def test_prediction_pipeline_on_real_image(self):
        import os
        import cv2
        from src.prediction import predict_emotions_in_frame

        test_img_path = 'camera_test.jpg'
        if os.path.exists(test_img_path):
            img = cv2.imread(test_img_path)
            result = predict_emotions_in_frame(img)
            self.assertEqual(result['faces_count'], 1)
            self.assertTrue(len(result['results']) > 0)
            face = result['results'][0]
            self.assertIn('emotion', face)
            self.assertIn('confidence', face)
            self.assertIn('is_uncertain', face)
            self.assertIn('scores', face)
            self.assertEqual(len(face['scores']), 7)
            # Top emotion must be one of the seven supported classes, never "Uncertain"
            expected_emotions = {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'}
            self.assertIn(face['emotion'], expected_emotions)
            self.assertNotEqual(face['emotion'].lower(), 'uncertain')
            # Check scores sum to ~100%
            total_score = sum(face['scores'].values())
            self.assertAlmostEqual(total_score, 100.0, delta=1.5)
            # Bounding box must be within image bounds
            x, y, w, h = face['box']
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, img.shape[1])
            self.assertLessEqual(y + h, img.shape[0])

    def test_model_loading_and_labels_mapping(self):
        from src.emotion_recognition import get_emotion_recognizer
        recognizer = get_emotion_recognizer()
        self.assertTrue(recognizer.is_model_loaded)
        self.assertEqual(len(recognizer.labels), 7)
        self.assertEqual(
            recognizer.labels,
            ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        )

    def test_uploaded_photo_face_detection(self):
        import os
        import cv2
        from src.face_detection import get_face_detector
        from src.prediction import predict_emotions_in_frame

        detector = get_face_detector()
        self.assertTrue(detector.is_available)
        self.assertIsNotNone(detector.face_cascade)

        # Check candidate test photo
        candidates = [
            r'C:\Users\User\Downloads\images (1).jpg',
            'camera_test.jpg',
            'camera_test_now.jpg'
        ]
        tested = False
        for path in candidates:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    faces = detector.detect_faces(img)
                    self.assertGreater(len(faces), 0, f"Face should be detected in {path}")
                    result = predict_emotions_in_frame(img)
                    self.assertGreater(result['faces_count'], 0)
                    self.assertGreater(len(result['results']), 0)
                    top = result['results'][0]
                    self.assertIn(top['emotion'], ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
                    self.assertGreater(top['confidence'], 0.0)
                    tested = True
                    break
        self.assertTrue(tested, "At least one test image must be evaluated")


