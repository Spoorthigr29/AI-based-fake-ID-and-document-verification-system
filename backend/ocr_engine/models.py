from django.db import models

class OCRAnalysis(models.Model):
    """
    Complete OCR execution record storing raw transcribed text,
    image quality metrics, engine diagnostics, and aggregate confidence.
    """
    document = models.OneToOneField(
        'documents.Document',
        on_delete=models.CASCADE,
        related_name='ocr_analysis'
    )
    raw_text = models.TextField(
        blank=True,
        help_text="Raw unformatted text transcribed from the document"
    )
    overall_confidence = models.FloatField(
        default=0.0,
        help_text="Average OCR token recognition confidence score (0.0 to 1.0)"
    )
    engine_used = models.CharField(
        max_length=50,
        default='PaddleOCR',
        help_text="OCR engine backend used for execution"
    )
    quality_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text="Sharpness, contrast, brightness, blur assessment"
    )
    deskew_angle = models.FloatField(
        default=0.0,
        help_text="Detected and corrected skew angle in degrees"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'OCR Analysis Record'
        verbose_name_plural = 'OCR Analysis Records'
        ordering = ['-created_at']

    def __str__(self):
        return f"OCR [{self.document.verification_id}] ({self.engine_used}, Conf: {self.overall_confidence:.2f})"

class ExtractedField(models.Model):
    """
    Key-value structured data points extracted from identity documents
    via OCR processing along with individual field confidence scores.
    """
    document = models.ForeignKey(
        'documents.Document',
        on_delete=models.CASCADE,
        related_name='extracted_fields'
    )
    field_name = models.CharField(
        max_length=100,
        help_text="Field identifier (e.g., name, date_of_birth, document_number, address, gender, issue_date, expiry_date, document_type)"
    )
    field_value = models.TextField(
        blank=True,
        help_text="Normalized textual value extracted from the document"
    )
    confidence = models.FloatField(
        default=0.0,
        help_text="Field-level extraction confidence (0.0 to 1.0)"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Extracted Field'
        verbose_name_plural = 'Extracted Fields'
        ordering = ['document', 'field_name']

    def __str__(self):
        return f"{self.document.verification_id} - {self.field_name}: {self.field_value[:30]} ({self.confidence:.2f})"
