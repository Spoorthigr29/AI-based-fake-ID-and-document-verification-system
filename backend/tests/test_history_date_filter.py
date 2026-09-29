import os
import sys
import unittest
from datetime import datetime, timedelta

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from django.utils import timezone
from django.test import RequestFactory
from django.contrib.auth.models import User, AnonymousUser

from documents.models import Document
from documents.views import verification_list_view, parse_date_str
from identity_verification.models import VerificationResult


class HistoryDateFilterTests(unittest.TestCase):
    """
    Test suite for Verification History Date-Range Filtering,
    Screened Document Counting, Summary metrics, and Pagination preservation.
    """

    @classmethod
    def setUpClass(cls):
        cls.factory = RequestFactory()

    def test_1_parse_date_str_various_formats(self):
        """Test parsing of DD-MM-YYYY, DD/MM/YYYY, and YYYY-MM-DD."""
        dt_start = parse_date_str("01-09-2026", is_end=False)
        self.assertIsNotNone(dt_start)
        self.assertEqual(dt_start.day, 1)
        self.assertEqual(dt_start.month, 9)
        self.assertEqual(dt_start.year, 2026)
        self.assertEqual(dt_start.hour, 0)
        self.assertEqual(dt_start.minute, 0)
        self.assertEqual(dt_start.second, 0)

        dt_end = parse_date_str("16-09-2026", is_end=True)
        self.assertIsNotNone(dt_end)
        self.assertEqual(dt_end.day, 16)
        self.assertEqual(dt_end.month, 9)
        self.assertEqual(dt_end.year, 2026)
        self.assertEqual(dt_end.hour, 23)
        self.assertEqual(dt_end.minute, 59)
        self.assertEqual(dt_end.second, 59)

        # Alternative formats
        dt_slash = parse_date_str("16/09/2026")
        self.assertIsNotNone(dt_slash)
        self.assertEqual(dt_slash.day, 16)

        dt_iso = parse_date_str("2026-09-16")
        self.assertIsNotNone(dt_iso)
        self.assertEqual(dt_iso.day, 16)

        print("[OK] Test 1 (Date Parsing) passed")

    def test_2_filter_active_date_range(self):
        """Test history view with matching date range."""
        # 26-09-2026 to 29-09-2026 covers existing records
        req = self.factory.get('/documents/history/?from_date=26-09-2026&to_date=29-09-2026')
        req.user = AnonymousUser()
        req.session = {}

        response = verification_list_view(req)
        self.assertEqual(response.status_code, 200)

        content = response.content.decode('utf-8')
        self.assertIn("Documents Screened:", content)
        self.assertIn("Verification Records", content)
        self.assertIn("Verified", content)
        self.assertIn("Review Required", content)
        self.assertIn("Mismatch", content)
        print("[OK] Test 2 (Filter Active Range) passed")

    def test_3_filter_empty_date_range(self):
        """Test history view with future date range containing zero records."""
        req = self.factory.get('/documents/history/?from_date=01-01-2030&to_date=10-01-2030')
        req.user = AnonymousUser()
        req.session = {}

        response = verification_list_view(req)
        self.assertEqual(response.status_code, 200)

        content = response.content.decode('utf-8')
        self.assertIn("No verification records found for the selected date range.", content)
        self.assertIn("Documents Screened: 0", content)
        print("[OK] Test 3 (Filter Empty Range message) passed")

    def test_4_clear_filter_returns_all_records(self):
        """Test unfiltered request returns all records."""
        # Test as admin / reviewer user who sees all 167 records
        admin_user, _ = User.objects.get_or_create(username="admin_test", defaults={"is_staff": True, "is_superuser": True})
        req = self.factory.get('/documents/history/')
        req.user = admin_user
        req.session = {}

        response = verification_list_view(req)
        self.assertEqual(response.status_code, 200)

        content = response.content.decode('utf-8')
        total_db = Document.objects.count()
        self.assertIn(f"Documents Screened: {total_db}", content)
        print("[OK] Test 4 (Clear Filter full count) passed")


if __name__ == '__main__':
    unittest.main()
