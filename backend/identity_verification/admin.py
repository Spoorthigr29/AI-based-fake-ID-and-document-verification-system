from django.contrib import admin
from .models import VerificationResult

@admin.register(VerificationResult)
class VerificationResultAdmin(admin.ModelAdmin):
    list_display = (
        'document',
        'overall_risk_score',
        'risk_category',
        'ocr_score',
        'face_score',
        'tamper_score',
        'consistency_score',
        'created_at'
    )
    list_filter = ('risk_category', 'created_at')
    search_fields = ('document__verification_id', 'explanation')
