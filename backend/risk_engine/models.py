from django.db import models

class RiskFactor(models.Model):
    """
    Granular explainable risk factor breakdown quantifying each signal's
    relative contribution to the aggregate risk calculation.
    """
    verification_result = models.ForeignKey(
        'identity_verification.VerificationResult',
        on_delete=models.CASCADE,
        related_name='risk_factors'
    )
    factor_name = models.CharField(
        max_length=150,
        help_text="Name of the risk factor (e.g., Font Inconsistency, Low Face Similarity, ELA Variance)"
    )
    factor_value = models.CharField(
        max_length=255,
        help_text="Measured factor value or detection metric"
    )
    contribution = models.FloatField(
        default=0.0,
        help_text="Contribution weight to the total risk score (percentage or points)"
    )
    description = models.TextField(
        blank=True,
        help_text="Human-readable explanation of why this factor was triggered"
    )

    class Meta:
        verbose_name = 'Risk Factor'
        verbose_name_plural = 'Risk Factors'
        ordering = ['-contribution']

    def __str__(self):
        return f"{self.factor_name} (+{self.contribution:.1f} pts)"
