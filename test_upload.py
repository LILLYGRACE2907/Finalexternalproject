import os, sys
sys.path.append('D:/FACIAL_EMOTION_RECOGNIZATION_SYSTEM')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'facial_emotion_project.settings')
import django
django.setup()
from django.test import Client
from django.core.files.uploadedfile import SimpleUploadedFile

client = Client(enforce_csrf_checks=False)
# Load image
with open('D:/FACIAL_EMOTION_RECOGNIZATION_SYSTEM/camera_test.jpg', 'rb') as f:
    img_data = f.read()
resp = client.post('/api/process_frame/', {'image': SimpleUploadedFile('test.jpg', img_data, content_type='image/jpeg')})
print('Status:', resp.status_code)
print('Content:', resp.content.decode('utf-8')[:500])
