from django.contrib import admin
from .models import OCRAnalysis, ExtractedField

@admin.register(OCRAnalysis)
class OCRAnalysisAdmin(admin.ModelAdmin):
    list_display = ('document', 'engine_used', 'overall_confidence', 'deskew_angle', 'created_at')
    list_filter = ('engine_used', 'created_at')
    search_fields = ('document__verification_id', 'raw_text')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(ExtractedField)
class ExtractedFieldAdmin(admin.ModelAdmin):
    list_display = ('document', 'field_name', 'field_value', 'confidence', 'created_at')
    list_filter = ('field_name', 'created_at')
    search_fields = ('document__verification_id', 'field_name', 'field_value')
