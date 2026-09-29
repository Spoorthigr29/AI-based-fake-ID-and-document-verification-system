from django.db import models
from django.conf import settings
from typing import Optional, Dict, Any


class AuditLog(models.Model):
    """
    Immutable compliance log recording all system events, document uploads,
    AI screening scans, reviewer notes, document access, and access control checks.
    """
    ACTION_UPLOAD = 'DOCUMENT_UPLOAD'
    ACTION_VERIFICATION_START = 'VERIFICATION_START'
    ACTION_AI_SCAN = 'AI_SCREENING_EXECUTED'
    ACTION_AI_SCREENING = ACTION_AI_SCAN
    ACTION_CASE_OPENED = 'CASE_OPENED'
    ACTION_MANUAL_REVIEW = 'MANUAL_OFFICER_REVIEW'
    ACTION_REVIEW_NOTE = 'REVIEW_NOTE_ADDED'
    ACTION_STATUS_CHANGE = 'STATUS_UPDATED'
    ACTION_DOCUMENT_ACCESS = 'DOCUMENT_ACCESSED'
    ACTION_UNAUTHORIZED_ATTEMPT = 'UNAUTHORIZED_ACCESS_ATTEMPT'
    ACTION_REPORT_EXPORT = 'REPORT_EXPORTED'
    ACTION_AUTH = 'USER_AUTHENTICATION'

    ACTION_CHOICES = [
        (ACTION_UPLOAD, 'USER uploaded document'),
        (ACTION_VERIFICATION_START, 'USER started verification'),
        (ACTION_AI_SCAN, 'AI analysis completed'),
        (ACTION_CASE_OPENED, 'Reviewer opened case'),
        (ACTION_MANUAL_REVIEW, 'Reviewer marked case for manual review'),
        (ACTION_REVIEW_NOTE, 'Reviewer added review notes'),
        (ACTION_STATUS_CHANGE, 'Verification status updated'),
        (ACTION_DOCUMENT_ACCESS, 'Document accessed'),
        (ACTION_UNAUTHORIZED_ATTEMPT, 'Unauthorized access attempted'),
        (ACTION_REPORT_EXPORT, 'Forensic report exported'),
        (ACTION_AUTH, 'User authentication event'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='audit_logs',
        help_text="User/Officer responsible for the action (or null for system actions)"
    )
    action = models.CharField(
        max_length=100,
        choices=ACTION_CHOICES,
        default=ACTION_AI_SCAN,
        help_text="Standardized audit action code"
    )
    verification_id = models.CharField(
        max_length=64,
        blank=True,
        db_index=True,
        help_text="Direct tracking ID for indexed lookup"
    )
    document = models.ForeignKey(
        'documents.Document',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='audit_logs',
        help_text="Associated document if applicable"
    )
    ip_address = models.GenericIPAddressField(
        null=True,
        blank=True,
        help_text="Originating IPv4 or IPv6 client address"
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Structured contextual metadata (IP, user agent, changes, scores)"
    )

    class Meta:
        verbose_name = 'Audit Log'
        verbose_name_plural = 'Audit Logs'
        ordering = ['-timestamp']

    @classmethod
    def log_event(
        cls,
        action: str,
        user: Optional[Any] = None,
        document: Optional[Any] = None,
        verification_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'AuditLog':
        """
        Standardized factory helper to record an immutable audit event.
        """
        v_id = verification_id
        if not v_id and document and hasattr(document, 'verification_id'):
            v_id = document.verification_id

        user_obj = user if (user and hasattr(user, 'is_authenticated') and user.is_authenticated) else None

        return cls.objects.create(
            user=user_obj,
            action=action,
            document=document,
            verification_id=v_id or '',
            ip_address=ip_address,
            metadata=metadata or {}
        )

    def __str__(self):
        actor = self.user.username if self.user else "SYSTEM"
        v_id = self.verification_id or (self.document.verification_id if self.document else "N/A")
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {actor} - {self.get_action_display()} (ID: {v_id})"
