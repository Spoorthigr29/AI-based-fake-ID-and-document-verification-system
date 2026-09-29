"""
Core unit and smoke tests for VERIFYX AI prototype.
"""
from django.test import TestCase
from django.urls import reverse
from documents.models import Document
from identity_verification.models import VerificationResult

class VerifyXSmokeTests(TestCase):
    def test_homepage_status(self):
        """Ensure homepage responds with 200 OK."""
        response = self.client.get(reverse('dashboard:home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'VERIFYX')
        self.assertContains(response, 'Start Verification')

    def test_dashboard_status(self):
        """Ensure dashboard view responds with 200 OK when authenticated."""
        from django.contrib.auth.models import User
        user = User.objects.create_user(username='testofficer', password='password123')
        self.client.login(username='testofficer', password='password123')
        response = self.client.get(reverse('dashboard:main'))
        self.assertEqual(response.status_code, 200)

    def test_document_creation(self):
        """Test creating a document instance generates auto verification_id."""
        doc = Document.objects.create(
            document_type=Document.DOC_TYPE_AADHAAR,
            processing_status=Document.STATUS_UPLOADED
        )
        self.assertTrue(doc.verification_id.startswith('VX-'))
        self.assertEqual(doc.processing_status, Document.STATUS_UPLOADED)
