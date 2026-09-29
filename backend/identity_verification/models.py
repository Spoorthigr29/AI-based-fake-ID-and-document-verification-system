from django.db import models

class VerificationResult(models.Model):
    """
    Master aggregated verification record combining OCR validity, biometric similarity,
    digital tampering detection, and cross-field algorithmic consistency.
    """
    CATEGORY_LOW_RISK = 'LOW_RISK'
    CATEGORY_MODERATE_RISK = 'MODERATE_RISK'
    CATEGORY_HIGH_RISK = 'HIGH_RISK'
    CATEGORY_MANUAL_REVIEW = 'SUSPICIOUS_MANUAL_REVIEW'

    RISK_CATEGORY_CHOICES = [
        (CATEGORY_LOW_RISK, 'Low Risk - Likely Authentic'),
        (CATEGORY_MODERATE_RISK, 'Moderate Risk - Flagged Signals'),
        (CATEGORY_HIGH_RISK, 'High Risk - High Probability Anomaly'),
        (CATEGORY_MANUAL_REVIEW, 'Suspicious - Officer Review Mandatory'),
    ]

    document = models.OneToOneField(
        'documents.Document',
        on_delete=models.CASCADE,
        related_name='verification_result'
    )
    overall_risk_score = models.FloatField(
        default=0.0,
        help_text="Aggregated risk index from 0.0 (clean) to 100.0 (high risk)"
    )
    risk_category = models.CharField(
        max_length=40,
        choices=RISK_CATEGORY_CHOICES,
        default=CATEGORY_LOW_RISK
    )
    ocr_score = models.FloatField(
        default=0.0,
        help_text="Text readability, format consistency, and checksum verification score (0.0 to 100.0)"
    )
    face_score = models.FloatField(
        default=0.0,
        help_text="Biometric confidence and facial match score (0.0 to 100.0)"
    )
    tamper_score = models.FloatField(
        default=0.0,
        help_text="Digital forensics integrity score (0.0 to 100.0, higher means cleaner image)"
    )
    consistency_score = models.FloatField(
        default=0.0,
        help_text="Logical validation score (DOB vs age, ID regex format, layout geometry)"
    )
    explanation = models.TextField(
        blank=True,
        help_text="Plain-language explainable AI audit summary of factors leading to the risk evaluation"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Verification Result'
        verbose_name_plural = 'Verification Results'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.document.verification_id} - Risk: {self.overall_risk_score:.1f} ({self.get_risk_category_display()})"


class IdentityConsistencyCheck(models.Model):
    """
    Field-by-field cross-consistency check comparing document OCR extractions
    with applicant intake data, secondary documents, or structural validation rules.
    """
    STATUS_CONSISTENT = 'CONSISTENT'
    STATUS_MANUAL_REVIEW = 'MANUAL_REVIEW'
    STATUS_DISCREPANCY = 'DISCREPANCY'

    STATUS_CHOICES = [
        (STATUS_CONSISTENT, 'Consistent / Matched'),
        (STATUS_MANUAL_REVIEW, 'Manual Review Recommended'),
        (STATUS_DISCREPANCY, 'Field Discrepancies Detected'),
    ]

    document = models.OneToOneField(
        'documents.Document',
        on_delete=models.CASCADE,
        related_name='consistency_check'
    )
    name_match = models.BooleanField(
        default=False,
        help_text="Whether normalized name comparison satisfies fuzzy matching threshold"
    )
    dob_match = models.BooleanField(
        default=False,
        help_text="Whether normalized date of birth matches"
    )
    document_number_match = models.BooleanField(
        default=False,
        help_text="Whether document identification number format and value are consistent"
    )
    address_match = models.BooleanField(
        default=False,
        help_text="Whether normalized address components match"
    )
    gender_match = models.BooleanField(
        default=True,
        help_text="Whether gender is consistent or unspecified"
    )
    consistency_score = models.IntegerField(
        default=0,
        help_text="Composite cross-field consistency score (0 to 100)"
    )
    status = models.CharField(
        max_length=30,
        choices=STATUS_CHOICES,
        default=STATUS_MANUAL_REVIEW
    )
    field_details = models.JSONField(
        default=dict,
        blank=True,
        help_text="Detailed field-by-field raw vs normalized values and similarity metrics"
    )
    signals = models.JSONField(
        default=list,
        blank=True,
        help_text="Forensic signals and discrepancy notes"
    )
    checklist = models.JSONField(
        default=list,
        blank=True,
        help_text="Checklist items (e.g. '✓ Name matched', '⚠ Address mismatch')"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Identity Consistency Check'
        verbose_name_plural = 'Identity Consistency Checks'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.document.verification_id} - Consistency: {self.consistency_score}/100 ({self.status})"
