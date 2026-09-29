from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """
    Profile model extending Django auth User with role-based access control.
    Roles:
    - ADMIN: View all cases, view audit logs, manage system and users
    - REVIEWER: View assigned/suspicious cases, review dossiers, add review notes
    - USER: Upload documents, view own verifications
    """
    ROLE_ADMIN = 'ADMIN'
    ROLE_REVIEWER = 'REVIEWER'
    ROLE_USER = 'USER'

    ROLE_CHOICES = [
        (ROLE_ADMIN, 'Administrator'),
        (ROLE_REVIEWER, 'Forensic Reviewer / Officer'),
        (ROLE_USER, 'Standard Applicant / User'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_USER)
    department = models.CharField(max_length=100, blank=True, default='Identity Screening Unit')
    assigned_count = models.IntegerField(default=0, help_text="Number of cases assigned for review")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
        ordering = ['-created_at']

    def is_admin(self) -> bool:
        return self.role == self.ROLE_ADMIN or self.user.is_superuser or self.user.is_staff

    def is_reviewer(self) -> bool:
        return self.role in [self.ROLE_REVIEWER, self.ROLE_ADMIN] or self.user.is_staff or self.user.is_superuser

    def is_regular_user(self) -> bool:
        return self.role == self.ROLE_USER

    def __str__(self):
        return f"{self.user.username} ({self.get_role_display()})"


class ReviewNote(models.Model):
    """
    Reviewer audit note appended to suspicious or manual-review cases.
    """
    ACTION_APPROVE = 'APPROVE'
    ACTION_REJECT = 'REJECT'
    ACTION_FURTHER_REVIEW = 'FURTHER_REVIEW'

    ACTION_CHOICES = [
        (ACTION_APPROVE, 'Recommend Approval / Cleared'),
        (ACTION_REJECT, 'Recommend Rejection / Anomaly Confirmed'),
        (ACTION_FURTHER_REVIEW, 'Request Secondary In-Person Review'),
    ]

    document = models.ForeignKey(
        'documents.Document',
        on_delete=models.CASCADE,
        related_name='review_notes'
    )
    author = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='authored_notes'
    )
    note = models.TextField(help_text="Detailed forensic observations and review rationale")
    recommended_action = models.CharField(
        max_length=30,
        choices=ACTION_CHOICES,
        default=ACTION_FURTHER_REVIEW
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Review Note'
        verbose_name_plural = 'Review Notes'
        ordering = ['-created_at']

    def __str__(self):
        return f"Note by {self.author.username} on {self.document.verification_id} ({self.recommended_action})"


@receiver(post_save, sender=User)
def create_or_save_user_profile(sender, instance, created, **kwargs):
    if created:
        role = UserProfile.ROLE_ADMIN if (instance.is_superuser or instance.is_staff) else UserProfile.ROLE_USER
        UserProfile.objects.create(user=instance, role=role)
    else:
        if hasattr(instance, 'profile'):
            instance.profile.save()
