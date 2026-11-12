import json
import base64
import cv2
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect, ensure_csrf_cookie

from .models import EmotionRecognition, ContactMessage
from .forms import RegisterForm, LoginForm, ContactForm
from src.utils import check_model_files, ensure_default_models
from src.database import get_user_dashboard_stats, get_user_recognition_history
from src.camera import VideoCamera, gen_frames
from src.prediction import predict_emotions_in_frame
from src.emotion_recognition import get_emotion_recognizer

# Global camera instance (lazily initialized)
_global_camera = None

def get_camera_instance():
    global _global_camera
    if _global_camera is None or not _global_camera.is_available:
        _global_camera = VideoCamera(0)
    return _global_camera


def home(request):
    """Landing Home Page"""
    model_ok, missing = check_model_files()
    context = {
        'title': 'Home - Facial Emotion Recognition System',
        'model_ok': model_ok,
    }
    return render(request, 'index.html', context)


def register_view(request):
    """User Registration View"""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            try:
                user = form.save()
                messages.success(request, "Account created successfully! Please log in with your credentials.")
                return redirect('login')
            except Exception as e:
                messages.error(request, f"Registration failed: {str(e)}")
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{error}")
    else:
        form = RegisterForm()

    return render(request, 'register.html', {'form': form, 'title': 'Register'})


def login_view(request):
    """User Login View"""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email_or_user = form.cleaned_data.get('email')
            password = form.cleaned_data.get('password')

            # Authenticate with username or email
            user = authenticate(request, username=email_or_user, password=password)
            if user is None:
                # Try finding user by email
                from django.contrib.auth.models import User
                try:
                    user_obj = User.objects.get(email=email_or_user)
                    user = authenticate(request, username=user_obj.username, password=password)
                except User.DoesNotExist:
                    user = None

            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.first_name or user.username}!")
                next_url = request.GET.get('next', 'dashboard')
                return redirect(next_url)
            else:
                messages.error(request, "Invalid email/username or password. Please try again.")
    else:
        form = LoginForm()

    return render(request, 'login.html', {'form': form, 'title': 'Login'})


def logout_view(request):
    """User Logout View"""
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('login')


@login_required
def dashboard(request):
    """User Dashboard View"""
    stats = get_user_dashboard_stats(request.user)
    recent_records = EmotionRecognition.objects.filter(user=request.user)[:10]

    context = {
        'title': 'Dashboard',
        'stats': stats,
        'recent_records': recent_records,
        'chart_labels': json.dumps(list(stats['emotion_distribution'].keys())),
        'chart_data': json.dumps(list(stats['emotion_distribution'].values())),
    }
    return render(request, 'dashboard.html', context)


@ensure_csrf_cookie
@login_required
def recognition(request):
    """Live Facial Emotion Recognition Page"""
    model_ok, missing_files = check_model_files()
    
    if not model_ok:
        ensure_default_models()
        model_ok, missing_files = check_model_files()

    recognizer = get_emotion_recognizer()
    emotions = recognizer.labels if recognizer and recognizer.labels else [
        'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'
    ]

    context = {
        'title': 'Facial Emotion Recognition',
        'model_ok': model_ok,
        'missing_files': missing_files,
        'emotions': emotions
    }
    return render(request, 'recognition.html', context)


@login_required
def history(request):
    """User Recognition History View with search & filtering"""
    selected_emotion = request.GET.get('emotion', 'all')
    search_query = request.GET.get('q', '').strip()
    sort_order = request.GET.get('sort', '-timestamp')

    queryset = EmotionRecognition.objects.filter(user=request.user)

    if selected_emotion and selected_emotion != 'all':
        queryset = queryset.filter(emotion__iexact=selected_emotion)

    if search_query:
        queryset = queryset.filter(emotion__icontains=search_query)

    records = queryset.order_by(sort_order)

    context = {
        'title': 'Recognition History',
        'records': records,
        'selected_emotion': selected_emotion,
        'search_query': search_query,
        'sort_order': sort_order,
        'emotions': ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    }
    return render(request, 'history.html', context)


def about(request):
    """About Page"""
    return render(request, 'about.html', {'title': 'About System'})


def contact(request):
    """Contact Page"""
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Thank you! Your message has been sent successfully. We will get back to you soon.")
            return redirect('contact')
        else:
            messages.error(request, "Please correct the errors in the form.")
    else:
        form = ContactForm()

    return render(request, 'contact.html', {'form': form, 'title': 'Contact Us'})


# Stream & API Views
@login_required
def video_feed(request):
    """Video streaming route for live webcam feed"""
    camera = get_camera_instance()
    return StreamingHttpResponse(
        gen_frames(camera),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@csrf_protect
def process_frame(request):
    # Allow unauthenticated calls but require authentication for saving results.
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Authentication required.'}, status=401)
    """API endpoint to process a single uploaded image or base64 frame"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST method required.'}, status=405)

    try:
        img_bytes = None

        # Case 1: Uploaded image file (multipart/form-data)
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            filename = (image_file.name or '').lower()
            valid_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
            if not any(filename.endswith(ext) for ext in valid_exts):
                return JsonResponse({
                    'success': False,
                    'error': 'Unsupported file format. Please upload a JPG, JPEG, PNG, WEBP, or BMP image.'
                }, status=400)

            if image_file.size > 10 * 1024 * 1024:
                return JsonResponse({
                    'success': False,
                    'error': 'File is too large. Maximum supported image size is 10MB.'
                }, status=400)

            img_bytes = image_file.read()

        # Case 2: Base64 data URI string or raw base64 (application/json)
        elif request.body:
            frame_b64 = ''
            try:
                body_json = json.loads(request.body.decode('utf-8'))
                frame_b64 = body_json.get('image', '')
            except (json.JSONDecodeError, UnicodeDecodeError):
                frame_b64 = request.body.decode('utf-8', errors='ignore')

            if ',' in frame_b64:
                frame_b64 = frame_b64.split(',', 1)[1]

            frame_b64 = frame_b64.strip()
            if frame_b64:
                missing_padding = len(frame_b64) % 4
                if missing_padding:
                    frame_b64 += '=' * (4 - missing_padding)

                try:
                    img_bytes = base64.b64decode(frame_b64)
                except Exception as decode_err:
                    return JsonResponse({
                        'success': False,
                        'error': f'Failed to decode image data: {str(decode_err)}'
                    }, status=400)

        if not img_bytes:
            return JsonResponse({'success': False, 'error': 'Invalid or empty image data received.'}, status=400)

        # Consistent decode with EXIF orientation handling (Pillow) with robust OpenCV fallback
        image_data = None
        try:
            from io import BytesIO
            from PIL import Image, ImageOps
            pil_img = Image.open(BytesIO(img_bytes))
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb_arr = np.array(pil_img.convert('RGB'))
            image_data = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception:
            np_arr = np.frombuffer(img_bytes, np.uint8)
            image_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_data is None or image_data.size == 0:
            return JsonResponse({'success': False, 'error': 'Could not decode image from provided data.'}, status=400)

        # Run prediction
        result_info = predict_emotions_in_frame(image_data)
        
        # If the model is unavailable, return 503 Service Unavailable
        if not result_info.get('model_loaded', False):
            return JsonResponse({
                'success': False,
                'error': result_info.get('model_error') or 'Emotion model is not available.',
                'model_loaded': False,
                'faces_count': result_info.get('faces_count', 0),
                'results': [],
            }, status=503)

        # Convert annotated frame to base64 for display
        annotated = result_info.get('annotated_frame')
        annotated_b64 = ""
        if annotated is not None:
            ret, buffer = cv2.imencode('.jpg', annotated)
            if ret:
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        return JsonResponse({
            'success': True,
            'faces_count': result_info.get('faces_count', 0),
            'results': result_info.get('results', []),
            'model_loaded': True,
            'model_error': None,
            'annotated_image': annotated_b64
        })
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("Unexpected exception in process_frame: %s", str(e))
        return JsonResponse({'success': False, 'error': f'Server error processing frame: {type(e).__name__}'}, status=500)


@csrf_protect
@login_required
def save_recognition(request):
    """Save recognition result for current user"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            emotion = data.get('emotion', '').strip()
            confidence = float(data.get('confidence', 0.0))

            if not emotion:
                return JsonResponse({'success': False, 'error': 'Emotion is required.'}, status=400)

            record = EmotionRecognition.objects.create(
                user=request.user,
                emotion=emotion,
                confidence=confidence
            )

            return JsonResponse({
                'success': True,
                'message': 'Recognition result saved successfully!',
                'record_id': record.id,
                'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)

    return JsonResponse({'success': False, 'error': 'POST method required.'}, status=405)


@login_required
def delete_history(request, pk):
    """Delete recognition record owned by user"""
    record = get_object_or_404(EmotionRecognition, pk=pk, user=request.user)
    record.delete()
    messages.success(request, "Recognition record deleted successfully.")
    return redirect('history')
