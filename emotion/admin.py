from django.contrib import admin
from .models import EmotionRecognition, ContactMessage

@admin.register(EmotionRecognition)
class EmotionRecognitionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'emotion', 'confidence_display', 'timestamp')
    list_filter = ('emotion', 'timestamp', 'user')
    search_fields = ('user__username', 'user__email', 'user__first_name', 'user__last_name', 'emotion')
    ordering = ('-timestamp',)
    readonly_fields = ('timestamp',)

    def confidence_display(self, obj):
        return f"{obj.confidence:.1f}%"
    confidence_display.short_description = 'Confidence'

@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'email', 'created_at', 'short_message')
    list_filter = ('created_at',)
    search_fields = ('name', 'email', 'message')
    ordering = ('-created_at',)
    readonly_fields = ('created_at',)

    def short_message(self, obj):
        return obj.message[:50] + "..." if len(obj.message) > 50 else obj.message
    short_message.short_description = 'Message Preview'
