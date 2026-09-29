"""
VerifyX AI - Demo History Commands Tests
========================================
Tests the generate_demo_history and clear_demo_history management commands.
"""

from io import StringIO
from django.core.management import call_command
from django.test import TestCase
from documents.models import Document


class DemoHistoryManagementCommandsTests(TestCase):
    def test_generate_and_clear_demo_history(self):
        """Test creating and clearing demo records while preserving real records."""
        # 1. Create a real record
        real_doc = Document.objects.create(
            verification_id="VX-2026-REAL01",
            document_type=Document.DOC_TYPE_AADHAAR,
            is_demo=False
        )

        out = StringIO()
        # 2. Call generate_demo_history with count=10
        call_command('generate_demo_history', count=10, stdout=out)
        self.assertIn("Successfully created 10 DEMO screening records", out.getvalue())

        demo_count = Document.objects.filter(is_demo=True).count()
        self.assertEqual(demo_count, 10)

        # 3. Ensure real record was untouched
        self.assertTrue(Document.objects.filter(verification_id="VX-2026-REAL01", is_demo=False).exists())

        # 4. Call clear_demo_history
        clear_out = StringIO()
        call_command('clear_demo_history', stdout=clear_out)
        self.assertIn("Successfully cleared 10 DEMO screening records", clear_out.getvalue())
        self.assertIn("Preserved 1 real screening records", clear_out.getvalue())

        # 5. Verify demo records are gone, real record preserved
        self.assertEqual(Document.objects.filter(is_demo=True).count(), 0)
        self.assertEqual(Document.objects.filter(is_demo=False).count(), 1)
