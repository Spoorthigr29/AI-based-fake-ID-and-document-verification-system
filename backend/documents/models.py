import hashlib
import uuid
from datetime import datetime
from django.db import models, transaction
from django.contrib.auth.models import User
from .validators import validate_uploaded_document, validate_uploaded_selfie

def document_upload_path(instance, filename):
    """
    Generate secure storage path for identity documents using randomized UUID.
    Avoids exposing original client filenames or allowing directory traversal attacks.
    """
    ext = filename.split('.')[-1].lower() if '.' in filename else 'jpg'
    date_str = datetime.now().strftime('%Y/%m/%d')
    return f"documents/{date_str}/{uuid.uuid4().hex}.{ext}"

def selfie_upload_path(instance, filename):
    """
    Generate secure storage path for applicant selfie photos using randomized UUID.
    """
    ext = filename.split('.')[-1].lower() if '.' in filename else 'jpg'
    date_str = datetime.now().strftime('%Y/%m/%d')
    return f"selfies/{date_str}/{uuid.uuid4().hex}.{ext}"

def generate_verification_id():
    """
    Generate sequential institutional verification ID in format VX-YYYY-XXXXXX (e.g. VX-2026-000001).
    """
    year = datetime.now().year
    prefix = f"VX-{year}-"
    
    with transaction.atomic():
        last_doc = Document.objects.filter(
            verification_id__startswith=prefix
        ).order_by('-verification_id').first()

        if last_doc:
            try:
                last_seq = int(last_doc.verification_id.split('-')[-1])
                new_seq = last_seq + 1
            except (ValueError, IndexError):
                new_seq = 1
        else:
            new_seq = 1

        new_id = f"{prefix}{new_seq:06d}"
        while Document.objects.filter(verification_id=new_id).exists():
            new_seq += 1
            new_id = f"{prefix}{new_seq:06d}"
            
        return new_id

class Document(models.Model):
    """
    Central Document Model storing identity document uploads, applicant selfies,
    cryptographic hashes for tamper integrity, and processing statuses.
    """
    DOC_TYPE_AADHAAR = 'AADHAAR'
    DOC_TYPE_PAN = 'PAN'
    DOC_TYPE_PASSPORT = 'PASSPORT'
    DOC_TYPE_DRIVING_LICENSE = 'DRIVING_LICENSE'
    DOC_TYPE_VOTER_ID = 'VOTER_ID'
    DOC_TYPE_OTHER = 'OTHER'

    DOCUMENT_TYPE_CHOICES = [
        (DOC_TYPE_AADHAAR, 'Aadhaar Card'),
        (DOC_TYPE_PAN, 'Permanent Account Number (PAN) Card'),
        (DOC_TYPE_PASSPORT, 'Passport'),
        (DOC_TYPE_DRIVING_LICENSE, 'Driving License'),
        (DOC_TYPE_VOTER_ID, 'Voter ID (EPIC)'),
        (DOC_TYPE_OTHER, 'Other Government Issued ID'),
    ]

    STATUS_UPLOADED = 'UPLOADED'
    STATUS_PROCESSING = 'PROCESSING'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_FAILED = 'FAILED'
    STATUS_MANUAL_REVIEW = 'MANUAL_REVIEW'

    STATUS_CHOICES = [
        (STATUS_UPLOADED, 'Uploaded'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
        (STATUS_MANUAL_REVIEW, 'Manual Review'),
    ]

    id = models.BigAutoField(primary_key=True)
    verification_id = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        blank=True,
        help_text="Institutional tracking identifier (e.g. VX-2026-000001)"
    )
    document_type = models.CharField(
        max_length=30,
        choices=DOCUMENT_TYPE_CHOICES,
        default=DOC_TYPE_OTHER
    )
    original_file = models.FileField(
        upload_to=document_upload_path,
        validators=[validate_uploaded_document],
        help_text="Uploaded identity document (JPG, JPEG, PNG, WEBP, PDF)"
    )
    selfie_file = models.FileField(
        upload_to=selfie_upload_path,
        validators=[validate_uploaded_selfie],
        null=True,
        blank=True,
        help_text="Uploaded applicant selfie image (JPG, JPEG, PNG, WEBP)"
    )
    file_hash = models.CharField(
        max_length=64,
        blank=True,
        help_text="SHA-256 cryptographic digest of document for tamper validation"
    )
    selfie_hash = models.CharField(
        max_length=64,
        blank=True,
        help_text="SHA-256 cryptographic digest of applicant selfie"
    )
    upload_timestamp = models.DateTimeField(auto_now_add=True)
    processing_status = models.CharField(
        max_length=30,
        choices=STATUS_CHOICES,
        default=STATUS_UPLOADED
    )
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='uploaded_documents'
    )
    is_demo = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Flag indicating whether this record is a demo/test screening entry"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Identity Document'
        verbose_name_plural = 'Identity Documents'
        ordering = ['-created_at']

    def calculate_hashes(self):
        """Calculate and store SHA-256 hashes for uploaded files."""
        updated = False
        if self.original_file and not self.file_hash:
            try:
                self.original_file.seek(0)
                sha256 = hashlib.sha256()
                for chunk in self.original_file.chunks():
                    sha256.update(chunk)
                self.file_hash = sha256.hexdigest()
                updated = True
            except Exception:
                pass

        if self.selfie_file and not self.selfie_hash:
            try:
                self.selfie_file.seek(0)
                sha256 = hashlib.sha256()
                for chunk in self.selfie_file.chunks():
                    sha256.update(chunk)
                self.selfie_hash = sha256.hexdigest()
                updated = True
            except Exception:
                pass
        return updated

    def save(self, *args, **kwargs):
        if not self.verification_id:
            self.verification_id = generate_verification_id()
        super().save(*args, **kwargs)
        if self.calculate_hashes():
            super().save(update_fields=['file_hash', 'selfie_hash'])

    def __str__(self):
        return f"{self.verification_id} - {self.get_document_type_display()} ({self.get_processing_status_display()})"

class DocumentQualityAnalysis(models.Model):
    """
    Document classification and quality assessment record.
    Stores resolution, blur, brightness, contrast, skew, readability, and issues checklist.
    """
    document = models.OneToOneField(
        Document,
        on_delete=models.CASCADE,
        related_name='quality_analysis'
    )
    classified_type = models.CharField(
        max_length=50,
        default='unknown_document',
        help_text="Classified document category (e.g. sample_identity_card, sample_pan_document, etc.)"
    )
    classification_confidence = models.FloatField(
        default=0.50,
        help_text="Classifier confidence probability (0.0 to 1.0)"
    )
    quality_score = models.IntegerField(
        default=100,
        help_text="Composite document quality score (0 to 100)"
    )
    issues = models.JSONField(
        default=list,
        blank=True,
        help_text="List of detected quality issues (blur, glare, skew, low res)"
    )
    checklist = models.JSONField(
        default=list,
        blank=True,
        help_text="List of passed verification criteria"
    )
    metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text="Underlying numeric metrics: blur, brightness, contrast, skew, resolution"
    )
    is_poor_quality = models.BooleanField(
        default=False,
        help_text="Flag indicating whether document quality mandates manual officer review"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Document Quality Analysis'
        verbose_name_plural = 'Document Quality Analyses'
        ordering = ['-created_at']

    def __str__(self):
        return f"Quality [{self.document.verification_id}] - {self.quality_score}/100 ({self.classified_type})"
