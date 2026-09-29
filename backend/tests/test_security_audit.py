import io
import hashlib
from PIL import Image, ImageDraw
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from accounts.models import UserProfile, ReviewNote
from documents.models import Document
from audit.models import AuditLog
from accounts.permissions import can_access_document


def create_sample_image(content="SAMPLE_ID_TEST"):
    image = Image.new('RGB', (400, 300), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), content, fill=(0, 0, 0))
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return SimpleUploadedFile("sample.jpg", byte_arr.getvalue(), content_type="image/jpeg")


class SecurityAndAuditTests(TestCase):
    """
    Test suite for VerifyX AI Security, Role-Based Access Control, Secure File Streaming,
    and Immutable Audit Logging.
    """

    def setUp(self):
        self.client = Client()

        # 1. Create standard USER
        self.user_a = User.objects.create_user(username='user_alice', password='Password123!', email='alice@test.com')
        self.profile_a, _ = UserProfile.objects.get_or_create(user=self.user_a, defaults={'role': UserProfile.ROLE_USER})
        self.profile_a.role = UserProfile.ROLE_USER
        self.profile_a.save()

        # 2. Create another standard USER
        self.user_b = User.objects.create_user(username='user_bob', password='Password123!', email='bob@test.com')
        self.profile_b, _ = UserProfile.objects.get_or_create(user=self.user_b, defaults={'role': UserProfile.ROLE_USER})
        self.profile_b.role = UserProfile.ROLE_USER
        self.profile_b.save()

        # 3. Create REVIEWER
        self.reviewer_user = User.objects.create_user(username='reviewer_carol', password='Password123!', email='carol@test.com')
        self.profile_reviewer, _ = UserProfile.objects.get_or_create(user=self.reviewer_user, defaults={'role': UserProfile.ROLE_REVIEWER})
        self.profile_reviewer.role = UserProfile.ROLE_REVIEWER
        self.profile_reviewer.save()

        # 4. Create ADMIN
        self.admin_user = User.objects.create_superuser(username='admin_david', password='Password123!', email='david@test.com')
        self.profile_admin, _ = UserProfile.objects.get_or_create(user=self.admin_user, defaults={'role': UserProfile.ROLE_ADMIN})
        self.profile_admin.role = UserProfile.ROLE_ADMIN
        self.profile_admin.save()

        # Create Alice's Document
        self.doc_file = create_sample_image("ALICE_ID_DATA")
        self.selfie_file = create_sample_image("ALICE_SELFIE")
        self.doc_a = Document.objects.create(
            document_type=Document.DOC_TYPE_AADHAAR,
            original_file=self.doc_file,
            selfie_file=self.selfie_file,
            uploaded_by=self.user_a,
            processing_status=Document.STATUS_COMPLETED
        )

    def test_user_roles_and_profile_methods(self):
        """Verify role helper methods on UserProfile."""
        self.assertTrue(self.profile_a.is_regular_user())
        self.assertFalse(self.profile_a.is_reviewer())
        self.assertFalse(self.profile_a.is_admin())

        self.assertTrue(self.profile_reviewer.is_reviewer())
        self.assertFalse(self.profile_reviewer.is_regular_user())

        self.assertTrue(self.profile_admin.is_admin())

    def test_sha256_hash_integrity_calculation(self):
        """Verify document calculates and stores valid SHA-256 digest."""
        self.doc_a.calculate_hashes()
        self.assertTrue(len(self.doc_a.file_hash) == 64)

    def test_authorized_user_can_access_own_document(self):
        """Alice can access her own verification detail page."""
        self.client.login(username='user_alice', password='Password123!')
        url = reverse('documents:detail', kwargs={'verification_id': self.doc_a.verification_id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_unauthorized_user_is_forbidden_from_viewing_others_document(self):
        """Bob attempting to view Alice's document receives 403 Forbidden and logs security audit."""
        self.client.login(username='user_bob', password='Password123!')
        url = reverse('documents:detail', kwargs={'verification_id': self.doc_a.verification_id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 403)

        # Audit log must record UNAUTHORIZED_ACCESS_ATTEMPT
        unauth_log = AuditLog.objects.filter(
            action=AuditLog.ACTION_UNAUTHORIZED_ATTEMPT,
            user=self.user_b,
            verification_id=self.doc_a.verification_id
        ).exists()
        self.assertTrue(unauth_log)

    def test_reviewer_can_access_any_case(self):
        """Reviewer Carol can open Alice's document and access is recorded in audit log."""
        self.client.login(username='reviewer_carol', password='Password123!')
        url = reverse('documents:detail', kwargs={'verification_id': self.doc_a.verification_id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

        # Audit log must record CASE_OPENED
        case_opened_log = AuditLog.objects.filter(
            action=AuditLog.ACTION_CASE_OPENED,
            user=self.reviewer_user,
            verification_id=self.doc_a.verification_id
        ).exists()
        self.assertTrue(case_opened_log)

    def test_admin_can_access_admin_dashboard_and_cases(self):
        """Admin David can view admin dashboard and inspection views."""
        self.client.login(username='admin_david', password='Password123!')
        url = reverse('dashboard:admin_dashboard')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_secure_file_streaming_authorization(self):
        """Secure file streaming enforces authorization checks."""
        # 1. Bob cannot stream Alice's document file
        self.client.login(username='user_bob', password='Password123!')
        file_url = reverse('documents:secure_file', kwargs={'verification_id': self.doc_a.verification_id, 'file_type': 'original'})
        response_bob = self.client.get(file_url)
        self.assertEqual(response_bob.status_code, 403)

        # 2. Alice can securely stream her document file
        self.client.login(username='user_alice', password='Password123!')
        response_alice = self.client.get(file_url)
        self.assertEqual(response_alice.status_code, 200)

        # 3. Reviewer can stream document file
        self.client.login(username='reviewer_carol', password='Password123!')
        response_carol = self.client.get(file_url)
        self.assertEqual(response_carol.status_code, 200)

    def test_reviewer_can_add_review_note_and_update_status(self):
        """Reviewer Carol can append review notes and update case triage status."""
        self.client.login(username='reviewer_carol', password='Password123!')
        note_url = reverse('documents:add_review_note', kwargs={'verification_id': self.doc_a.verification_id})

        response = self.client.post(note_url, {
            'note': 'Hologram and signature checked manually under magnifying loupe. Approved.',
            'recommended_action': ReviewNote.ACTION_APPROVE
        })
        self.assertEqual(response.status_code, 302)

        # Verify Note in DB
        note = ReviewNote.objects.filter(document=self.doc_a, author=self.reviewer_user).first()
        self.assertIsNotNone(note)
        self.assertEqual(note.recommended_action, ReviewNote.ACTION_APPROVE)

        # Verify Audit Log
        note_audit = AuditLog.objects.filter(
            action=AuditLog.ACTION_REVIEW_NOTE,
            user=self.reviewer_user,
            verification_id=self.doc_a.verification_id
        ).exists()
        self.assertTrue(note_audit)

    def test_regular_user_cannot_add_review_notes(self):
        """Regular user Alice cannot submit review notes."""
        self.client.login(username='user_alice', password='Password123!')
        note_url = reverse('documents:add_review_note', kwargs={'verification_id': self.doc_a.verification_id})

        response = self.client.post(note_url, {
            'note': 'Unauthorized note attempt',
            'recommended_action': ReviewNote.ACTION_APPROVE
        })
        # Role check returns 403 Forbidden
        self.assertEqual(response.status_code, 403)
