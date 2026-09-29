from django.db import models

class TamperAnalysis(models.Model):
    """
    Forensic image analysis results capturing Error Level Analysis (ELA),
    copy-move clone detection, boundary anomalies, and compression artifacts.
    """
    RISK_LOW = 'LOW'
    RISK_MODERATE = 'MODERATE'
    RISK_HIGH = 'HIGH'

    RISK_LEVEL_CHOICES = [
        (RISK_LOW, 'Low Risk (Clean / Consistent)'),
        (RISK_MODERATE, 'Moderate Risk (Minor Anomalies Detected)'),
        (RISK_HIGH, 'High Risk (Suspicious Tampering Signals)'),
    ]

    document = models.OneToOneField(
        'documents.Document',
        on_delete=models.CASCADE,
        related_name='tamper_analysis'
    )
    tampering_probability = models.FloatField(
        default=0.0,
        help_text="Calculated probability of digital manipulation (0.00 to 1.00)"
    )
    risk_level = models.CharField(
        max_length=20,
        choices=RISK_LEVEL_CHOICES,
        default=RISK_LOW,
        help_text="Risk level categorization (LOW, MODERATE, HIGH)"
    )
    signals = models.JSONField(
        default=list,
        blank=True,
        help_text="List of explainable forensic signals / defect indicators identified"
    )
    suspicious_regions = models.JSONField(
        default=list,
        blank=True,
        help_text="Bounding boxes and pixel coordinates of detected anomalous zones"
    )
    annotated_image = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Relative media path to annotated forensic heatmap / bounding box visualization"
    )
    ela_image = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Relative media path to Error Level Analysis difference image"
    )
    metadata_findings = models.JSONField(
        default=dict,
        blank=True,
        help_text="Findings from EXIF, software tag, and intake metadata inspection"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Tamper Analysis'
        verbose_name_plural = 'Tamper Analyses'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.document.verification_id} - {self.risk_level} ({self.tampering_probability * 100:.1f}%)"
