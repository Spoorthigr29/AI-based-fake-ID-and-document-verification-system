from django.db import models

class FaceVerification(models.Model):
    """
    Biometric face comparison result comparing document photo with live selfie.
    Stores detection flags, counts, similarity scores, confidence, and explainable signals.
    """
    MATCH_STATUS_MATCH = 'MATCH'
    MATCH_STATUS_MISMATCH = 'MISMATCH'
    MATCH_STATUS_MANUAL_REVIEW = 'MANUAL_REVIEW'
    MATCH_STATUS_NO_FACE = 'NO_FACE'
    MATCH_STATUS_MULTI_FACE = 'MULTI_FACE'
    MATCH_STATUS_POOR_QUALITY = 'POOR_QUALITY'
    MATCH_STATUS_NOT_APPLICABLE = 'NOT_APPLICABLE'
    MATCH_STATUS_NOT_PERFORMED = 'NOT_PERFORMED'
    MATCH_STATUS_ERROR = 'ERROR'

    LIVENESS_PASSED = 'PASSED'
    LIVENESS_FAILED = 'FAILED'
    LIVENESS_INCONCLUSIVE = 'INCONCLUSIVE'
    LIVENESS_NOT_APPLICABLE = 'NOT_APPLICABLE'

    MATCH_STATUS_CHOICES = [
        (MATCH_STATUS_MATCH, 'Match Confirmed'),
        (MATCH_STATUS_MISMATCH, 'Face Discrepancy / Mismatch'),
        (MATCH_STATUS_MANUAL_REVIEW, 'Manual Review Required'),
        (MATCH_STATUS_NO_FACE, 'No Face Detected'),
        (MATCH_STATUS_MULTI_FACE, 'Multiple Faces Detected in Document'),
        (MATCH_STATUS_POOR_QUALITY, 'Poor Face Quality / Unusable'),
        (MATCH_STATUS_NOT_APPLICABLE, 'Not Applicable (No Document Photo)'),
        (MATCH_STATUS_NOT_PERFORMED, 'Not Performed'),
        (MATCH_STATUS_ERROR, 'Processing Error'),
    ]

    LIVENESS_STATUS_CHOICES = [
        (LIVENESS_PASSED, 'Liveness Passed'),
        (LIVENESS_FAILED, 'Liveness Failed / Spoof Detected'),
        (LIVENESS_INCONCLUSIVE, 'Inconclusive Liveness'),
        (LIVENESS_NOT_APPLICABLE, 'Not Applicable (No Selfie)'),
    ]

    document = models.OneToOneField(
        'documents.Document',
        on_delete=models.CASCADE,
        related_name='face_verification'
    )
    document_face_detected = models.BooleanField(
        default=False,
        help_text="Whether exactly one usable face was detected in the identity document"
    )
    selfie_face_detected = models.BooleanField(
        default=False,
        help_text="Whether a usable face was detected in the applicant selfie"
    )
    document_faces_count = models.IntegerField(
        default=0,
        help_text="Total number of face instances detected in the identity document"
    )
    selfie_faces_count = models.IntegerField(
        default=0,
        help_text="Total number of face instances detected in the applicant selfie"
    )
    similarity_score = models.FloatField(
        null=True,
        blank=True,
        default=0.0,
        help_text="Cosine similarity score (0.00 to 1.00) or null if not applicable"
    )
    confidence = models.FloatField(
        null=True,
        blank=True,
        default=0.0,
        help_text="Overall biometric verification confidence (0.00 to 1.00) or null if not applicable"
    )
    match_status = models.CharField(
        max_length=30,
        choices=MATCH_STATUS_CHOICES,
        default=MATCH_STATUS_MANUAL_REVIEW,
        help_text="Verification determination status (MATCH, MANUAL_REVIEW, NOT_APPLICABLE, etc.)"
    )
    liveness_status = models.CharField(
        max_length=30,
        choices=LIVENESS_STATUS_CHOICES,
        default=LIVENESS_PASSED,
        help_text="Anti-spoofing live selfie assessment (PASSED, FAILED, INCONCLUSIVE, NOT_APPLICABLE)"
    )
    document_face_crop = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Relative media path to extracted document face crop"
    )
    selfie_face_crop = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Relative media path to extracted selfie face crop"
    )
    face_quality_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text="Detailed quality metrics for document and selfie face crops"
    )
    issues = models.JSONField(
        default=list,
        blank=True,
        help_text="List of issues, defects, or warning flags identified during face analysis"
    )
    explanation = models.TextField(
        blank=True,
        default="",
        help_text="Transparent explanation of face comparison or reason for non-applicability"
    )
    threshold_used = models.FloatField(
        default=0.70,
        help_text="Match threshold applied for this verification decision"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Face Verification Result'
        verbose_name_plural = 'Face Verification Results'
        ordering = ['-created_at']

    def __str__(self):
        sim_str = f"{self.similarity_score:.2f}" if self.similarity_score is not None else "N/A"
        conf_str = f"{self.confidence:.2f}" if self.confidence is not None else "N/A"
        return f"{self.document.verification_id} - {self.match_status} (Sim: {sim_str}, Conf: {conf_str})"

