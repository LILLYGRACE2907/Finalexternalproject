from django.db import models
from django.contrib.auth.models import User

class EmotionRecognition(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='emotion_records')
    emotion = models.CharField(max_length=50)
    confidence = models.FloatField(help_text="Confidence percentage (0-100)")
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Emotion Recognition Record'
        verbose_name_plural = 'Emotion Recognition Records'

    def __str__(self):
        return f"{self.user.username} - {self.emotion} ({self.confidence:.1f}%)"

class ContactMessage(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Contact Message'
        verbose_name_plural = 'Contact Messages'

    def __str__(self):
        return f"Message from {self.name} ({self.email})"
