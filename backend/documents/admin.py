from django.contrib import admin
from .models import Document, DocumentQualityAnalysis

@admin.register(DocumentQualityAnalysis)
class DocumentQualityAnalysisAdmin(admin.ModelAdmin):
    list_display = ('document', 'classified_type', 'quality_score', 'is_poor_quality', 'created_at')
    list_filter = ('classified_type', 'is_poor_quality', 'created_at')
    search_fields = ('document__verification_id', 'classified_type')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('verification_id', 'document_type', 'processing_status', 'file_hash', 'upload_timestamp', 'uploaded_by')
    list_filter = ('document_type', 'processing_status', 'upload_timestamp')
    search_fields = ('verification_id', 'file_hash', 'uploaded_by__username')
    readonly_fields = ('verification_id', 'file_hash', 'upload_timestamp', 'created_at', 'updated_at')
