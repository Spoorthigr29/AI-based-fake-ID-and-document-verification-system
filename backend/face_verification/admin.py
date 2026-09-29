from django.contrib import admin
from .models import FaceVerification

@admin.register(FaceVerification)
class FaceVerificationAdmin(admin.ModelAdmin):
    list_display = ('document', 'match_status', 'similarity_score', 'confidence', 'created_at')
    list_filter = ('match_status', 'created_at')
    search_fields = ('document__verification_id',)
