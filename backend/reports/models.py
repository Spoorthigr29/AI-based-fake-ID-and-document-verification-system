from django.db import models
from django.conf import settings

class VerificationReport(models.Model):
    """
    Generated forensic and verification report records for auditing and compliance exports.
    """
    FORMAT_PDF = 'PDF'
    FORMAT_JSON = 'JSON'
    FORMAT_CSV = 'CSV'

    FORMAT_CHOICES = [
        (FORMAT_PDF, 'PDF Dossier'),
        (FORMAT_JSON, 'JSON Machine-Readable'),
        (FORMAT_CSV, 'CSV Export'),
    ]

    document = models.ForeignKey(
        'documents.Document',
        on_delete=models.CASCADE,
        related_name='reports'
    )
    report_format = models.CharField(max_length=10, choices=FORMAT_CHOICES, default=FORMAT_PDF)
    report_file = models.FileField(upload_to='reports/%Y/%m/', null=True, blank=True)
    generated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Verification Report'
        verbose_name_plural = 'Verification Reports'
        ordering = ['-created_at']

    def __str__(self):
        return f"Report {self.id} for {self.document.verification_id} ({self.report_format})"
