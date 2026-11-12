# FACIAL EMOTION RECOGNITION SYSTEM

An AI-powered, real-time facial emotion recognition web application built using **Django, Python, OpenCV, TensorFlow/Keras, NumPy, and SQLite**.

Designed for Diploma Projects, GitHub Portfolios, and AI Software Demonstrations.

---

## 🌟 Key Features

- **Live Webcam Recognition:** Real-time facial emotion analysis with bounding box overlay and confidence scoring.
- **7 Emotion Classes:** Detects Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **User Authentication:** Registration, login, password hashing, and user-specific data management.
- **Analytics Dashboard:** Displays recognition statistics and emotion distribution.
- **Personalized History:** User-specific recognition history with search, filtering, sorting, and deletion.
- **Contact & Support:** Integrated contact messaging system.
- **Responsive UI:** Modern interface built using HTML, CSS, JavaScript, and Bootstrap.

---

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Backend Framework:** Django
- **Computer Vision:** OpenCV
- **Deep Learning:** TensorFlow / Keras
- **Model:** Convolutional Neural Network (CNN)
- **Data Processing:** NumPy, Pillow, h5py
- **Database:** SQLite
- **Frontend:** HTML5, CSS3, JavaScript
- **UI Framework:** Bootstrap 5

---

## 📁 Project Architecture

```text
FACIAL_EMOTION_RECOGNIZATION_SYSTEM/
│
├── manage.py
│
├── facial_emotion_project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── emotion/
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   ├── views.py
│   └── migrations/
│
├── models/
│   ├── emotion_model.h5
│   ├── emotion_model_improved.keras
│   ├── emotion_model_trained.keras
│   ├── face_detector.xml
│   ├── emotion_labels.txt
│   └── emotion_labels_trained.txt
│
├── src/
│   ├── camera.py
│   ├── database.py
│   ├── emotion_recognition.py
│   ├── face_detection.py
│   ├── prediction.py
│   ├── preprocessing.py
│   └── utils.py
│
├── training/
│   ├── data_preparation.py
│   ├── evaluate_model.py
│   ├── improve_model.py
│   ├── train_model.py
│   ├── train_real_model.py
│   └── train_v2_model.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── recognition.html
│   ├── history.html
│   ├── about.html
│   └── contact.html
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/LILLYGRACE2907/Finalexternalproject.git
```

### 2. Open the Project Folder

```bash
cd Finalexternalproject
```

### 3. Create a Virtual Environment

```bash
python -m venv py312env
```

### 4. Activate Virtual Environment

Windows CMD:

```cmd
py312env\Scripts\activate.bat
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run Database Migrations

```bash
python manage.py migrate
```

### 7. Create Admin Account

```bash
python manage.py createsuperuser
```

### 8. Run Django Development Server

```bash
python manage.py runserver
```

Open your browser and visit:

**http://127.0.0.1:8000/**

---

## 📷 How to Use Emotion Recognition

1. Open the web application.
2. Register a new account or log in.
3. Navigate to the Emotion Recognition page.
4. Start the webcam or upload an image.
5. Allow camera access when requested.
6. View the detected face and predicted emotion.
7. Save recognition results.
8. View previous results in the History page.

---

## 🎯 Project Objectives

- Detect human faces from images and live camera input.
- Preprocess facial images for emotion classification.
- Recognize seven basic facial emotions using a CNN model.
- Integrate the trained model with Django.
- Provide real-time facial emotion recognition.
- Store and display recognition history.

---

## 🎓 Project Information

**Project Title:** Facial Emotion Recognition System

**Domain:** Artificial Intelligence and Machine Learning

**Programming Language:** Python

**Framework:** Django

**Deep Learning Model:** CNN

**Database:** SQLite

**License:** MIT License

**Status:** Developed for Academic Project Demonstration

---

## 👩‍💻 Author

**P. Lilly Grace**

Diploma in Communication and Computer Networking (CCN)

Aditya Polytechnic College

---

## 📌 Note

This project is developed for educational and demonstration purposes. Facial emotion predictions are model-based estimates and should not be treated as definitive measurements of a person's actual emotional state.
