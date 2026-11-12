"""
Database helper module leveraging Django ORM for saving and retrieving emotion recognition data.
"""
from django.db.models import Count, Avg, Max
from emotion.models import EmotionRecognition, ContactMessage

def save_recognition_result(user, emotion, confidence):
    """
    Save emotion recognition record for a user via Django ORM.
    """
    if not user or not user.is_authenticated:
        raise ValueError("User must be authenticated to save recognition results.")
    
    record = EmotionRecognition.objects.create(
        user=user,
        emotion=emotion,
        confidence=float(confidence)
    )
    return record

def get_user_recognition_history(user, emotion_filter=None, sort_order='-timestamp'):
    """
    Retrieve emotion history for the logged-in user.
    """
    if not user or not user.is_authenticated:
        return EmotionRecognition.objects.none()

    queryset = EmotionRecognition.objects.filter(user=user)
    
    if emotion_filter and emotion_filter != 'all':
        queryset = queryset.filter(emotion__iexact=emotion_filter)
        
    return queryset.order_by(sort_order)

def get_user_dashboard_stats(user):
    """
    Compute dashboard analytics for the specified user.
    """
    if not user or not user.is_authenticated:
        return {
            'total_sessions': 0,
            'most_detected': 'None',
            'latest_emotion': 'None',
            'avg_confidence': 0.0,
            'emotion_distribution': {}
        }

    user_records = EmotionRecognition.objects.filter(user=user)
    total_sessions = user_records.count()

    if total_sessions == 0:
        return {
            'total_sessions': 0,
            'most_detected': 'None',
            'latest_emotion': 'None',
            'avg_confidence': 0.0,
            'emotion_distribution': {}
        }

    # Latest emotion record
    latest = user_records.first()
    latest_emotion = f"{latest.emotion} ({latest.confidence:.1f}%)"

    # Average confidence
    avg_conf = user_records.aggregate(avg=Avg('confidence'))['avg'] or 0.0

    # Most detected emotion
    emotion_counts = user_records.values('emotion').annotate(count=Count('emotion')).order_by('-count')
    most_detected = emotion_counts[0]['emotion'] if emotion_counts else 'None'

    # Emotion distribution for Chart.js
    distribution = {item['emotion']: item['count'] for item in emotion_counts}

    return {
        'total_sessions': total_sessions,
        'most_detected': most_detected,
        'latest_emotion': latest_emotion,
        'avg_confidence': round(avg_conf, 1),
        'emotion_distribution': distribution
    }
