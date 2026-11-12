from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('recognition/', views.recognition, name='recognition'),
    path('history/', views.history, name='history'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    
    # Live streaming & API endpoints
    path('video_feed/', views.video_feed, name='video_feed'),
    path('api/process_frame/', views.process_frame, name='process_frame'),
    path('api/save_recognition/', views.save_recognition, name='save_recognition'),
    path('api/delete_history/<int:pk>/', views.delete_history, name='delete_history'),
]
