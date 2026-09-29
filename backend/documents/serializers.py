from rest_framework import serializers
from .models import Document
from .validators import validate_uploaded_document, validate_uploaded_selfie

class DocumentUploadSerializer(serializers.ModelSerializer):
    original_file = serializers.FileField(validators=[validate_uploaded_document])
    selfie_file = serializers.FileField(validators=[validate_uploaded_selfie], required=False, allow_null=True)

    class Meta:
        model = Document
        fields = [
            'id',
            'verification_id',
            'document_type',
            'original_file',
            'selfie_file',
            'file_hash',
            'selfie_hash',
            'processing_status',
            'upload_timestamp',
        ]
        read_only_fields = ['id', 'verification_id', 'file_hash', 'selfie_hash', 'processing_status', 'upload_timestamp']
