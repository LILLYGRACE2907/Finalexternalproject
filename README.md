# FACIAL EMOTION RECOGNITION SYSTEM

An AI-powered, real-time facial emotion recognition web application built using **Django**, **OpenCV**, **TensorFlow/Keras/PyTorch**, **NumPy**, and **SQLite**.

Designed for **Diploma Projects**, **GitHub Portfolios**, and **Real-World AI Software Demonstrations**.

---

## 🌟 Key Features

* 🎥 **Live Webcam Recognition**: Real-time facial emotion analysis with bounding box overlay and confidence scoring.
* 🤖 **7 Emotion Classes**: Detects *Angry*, *Disgust*, *Fear*, *Happy*, *Sad*, *Surprise*, and *Neutral*.
* 👤 **User Authentication**: Secure registration, login, password hashing, and user-specific data isolation using Django ORM.
* 📊 **Analytics Dashboard**: Interactive stats displaying total sessions, top emotion, mean confidence, and Chart.js emotion distribution.
* 📜 **Personalized History**: User-specific history logs with live search, emotion filtering, sorting, and session deletion.
* 📩 **Contact & Support**: Integrated contact messaging system managed via Django Admin.
* 📱 **Responsive Modern UI**: Built with Bootstrap 5, Bootstrap Icons, dark AI technology theme, and CSS animations.

---

## 🛠️ Technology Stack

* **Backend Framework**: Django 5.x (Python)
* **Computer Vision**: OpenCV (Haar Cascade Face Detector)
* **Deep Learning**: TensorFlow / Keras / PyTorch (CNN Classifier)
* **Data Processing**: NumPy, Pillow, h5py
* **Database**: SQLite3 (via Django ORM)
* **Frontend**: HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Chart.js

---

## 📁 Project Architecture

```
FACIAL_EMOTION_RECOGNIZATION_SYSTEM/
│
├── manage.py                          # Django management script
│
├── facial_emotion_project/            # Main Django project package
│   ├── __init__.py
│   ├── settings.py                    # Project configuration & settings
│   ├── urls.py                        # Main URL routing
│   ├── asgi.py                        # ASGI config
│   └── wsgi.py                        # WSGI config
│
├── emotion/                           # Core Django application app
│   ├── admin.py                       # Admin panel registrations
│   ├── apps.py                        # App configuration
│   ├── models.py                      # EmotionRecognition & ContactMessage ORM models
│   ├── urls.py                        # Emotion app routing
│   ├── views.py                        # Page, authentication & API views
│   ├── forms.py                       # Registration, login & contact forms
│   └── migrations/                    # Database migrations
│
├── dataset/                           # Dataset storage
│   ├── train/                         # Training data by emotion category
│   └── test/                          # Test evaluation data
│
├── models/                            # AI Model assets
│   ├── emotion_model.h5               # Trained CNN emotion model weights
│   ├── face_detector.xml              # OpenCV Haar Cascade XML classifier
│   └── emotion_labels.txt             # Emotion class labels
│
├── src/                               # Core AI & computer vision logic
│   ├── __init__.py
│   ├── face_detection.py              # OpenCV face detection module
│   ├── emotion_recognition.py         # CNN model loading & inference engine
│   ├── camera.py                      # Video capture & web streaming generator
│   ├── preprocessing.py               # Grayscale, resize, normalization pipeline
│   ├── prediction.py                  # High-level frame inference interface
│   ├── database.py                    # Django ORM database helper routines
│   └── utils.py                       # Helper utilities & model file checks
│
├── training/                          # Model training & evaluation pipeline
│   ├── data_preparation.py            # Dataset structure preparation script
│   ├── train_model.py                 # CNN training script
│   └── evaluate_model.py              # Model accuracy & evaluation report generator
│
├── static/                            # Static asset directory
│   ├── css/style.css                  # Custom styling
│   └── js/script.js                   # Interactivity & helper scripts
│
├── templates/                         # HTML templates
│   ├── base.html                      # Main layout template
│   ├── index.html                     # Landing page
│   ├── login.html                     # Login page
│   ├── register.html                  # User registration page
│   ├── dashboard.html                 # Analytics dashboard
│   ├── recognition.html               # Real-time webcam recognition page
│   ├── history.html                   # User recognition history
│   ├── about.html                     # System documentation page
│   └── contact.html                   # Contact form page
│
├── results/                           # Experiment outputs & reports
│   └── reports/                       # Evaluation JSON reports
│
├── requirements.txt                   # Project dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Version control exclusions
```

---

## ⚙️ Installation & Setup Guide

### 1. Clone or Open Project
Navigate into the root directory:
```bash
cd FACIAL_EMOTION_RECOGNIZATION_SYSTEM
```

### 2. Activate Virtual Environment
```bash
# Windows (PowerShell / CMD)
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Database Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Create Superuser (Admin Access)
```bash
python manage.py createsuperuser
```
*(Or use default seeded credentials: Username `admin` / Password `adminpassword123`)*

### 6. Generate AI Model Files
```bash
python training/train_model.py
```

### 7. Launch Django Development Server
```bash
python manage.py runserver
```

Open your browser and visit:
👉 **`http://127.0.0.1:8000/`**

---

## 🔑 Demo Account Credentials

* **Admin User**: `admin` / Password: `adminpassword123`
* **Demo User**: `demo@example.com` / Password: `demopassword123`

---

## 📷 How to Use Live Emotion Recognition

1. Log in to your account.
2. Click **Start Recognition** in the navigation bar.
3. Allow browser webcam access or upload a photo.
4. Click **Capture & Analyze** or view the live bounding box annotations.
5. Click **Save Result** to record your session to your personal history!

---

## 🎓 Diploma & GitHub Project Details

* **Framework**: Django 5 Architecture
* **License**: MIT License
* **Status**: Complete & Production Ready
