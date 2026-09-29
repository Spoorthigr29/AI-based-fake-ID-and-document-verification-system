import os
import sys
import unittest
from typing import Dict, Any

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from rag.services.rag_service import RAGService
from rag.services.vector_store import LocalVectorStore
from rag.services.retriever import RAGRetriever
from rag.services.document_loader import DocumentLoader
from rag.services.text_chunker import TextChunker


class RAGPipelineTests(unittest.TestCase):
    """
    Test suite verifying RAG retrieval, rule matching, knowledge synthesis,
    and graceful error handling across all required scenarios.
    """

    @classmethod
    def setUpClass(cls):
        cls.rag_service = RAGService.get_instance()
        cls.rag_service.initialize(force_rebuild=True)

    def test_1_aadhaar_document_retrieval(self):
        """1. Test Aadhaar document retrieval and knowledge extraction."""
        extracted_text = "Government of India Unique Identification Authority of India 1234 5678 9012 DOB: 01/01/1990"
        analysis_results = {
            "overall_risk_score": 12,
            "risk_category": "LOW_RISK",
            "doc_quality_val": 95,
            "ocr_conf_val": 94,
            "tamper_risk_val": 8,
            "qr_status_display": "Verified (Secure QR)",
            "face_match_status": "MATCH",
            "face_match_val": 92
        }

        res = self.rag_service.generate_explanation(
            document_type="AADHAAR",
            extracted_text=extracted_text,
            analysis_results=analysis_results
        )

        self.assertEqual(res["document_type"], "AADHAAR")
        self.assertGreaterEqual(len(res["retrieved_context"]), 1)
        self.assertGreaterEqual(len(res["evidence_found"]), 1)
        self.assertGreaterEqual(len(res["relevant_rules"]), 1)
        self.assertIn("Aadhaar", res["explanation"])
        self.assertIn("Low", res["explanation"])
        print("[OK] Test 1 (Aadhaar retrieval) passed")

    def test_2_pan_document_retrieval(self):
        """2. Test PAN document retrieval and knowledge extraction."""
        extracted_text = "INCOME TAX DEPARTMENT GOVT. OF INDIA ABCDE1234F Permanent Account Number"
        analysis_results = {
            "overall_risk_score": 15,
            "risk_category": "LOW_RISK",
            "doc_quality_val": 90,
            "ocr_conf_val": 91,
            "tamper_risk_val": 10,
            "is_face_applicable": False,
            "face_match_status": "NOT_APPLICABLE"
        }

        res = self.rag_service.generate_explanation(
            document_type="PAN",
            extracted_text=extracted_text,
            analysis_results=analysis_results
        )

        self.assertEqual(res["document_type"], "PAN")
        self.assertGreaterEqual(len(res["retrieved_context"]), 1)
        # Check that PAN-specific context was retrieved
        context_snippets = " ".join([c["snippet"] for c in res["retrieved_context"]]).upper()
        self.assertTrue("PAN" in context_snippets or "INCOME TAX" in context_snippets)
        self.assertIn("PAN", res["explanation"])
        print("[OK] Test 2 (PAN retrieval) passed")

    def test_3_no_matching_knowledge(self):
        """3. Test retrieval behavior when document type has no specific reference in KB."""
        extracted_text = "Generic document snippet"
        analysis_results = {
            "overall_risk_score": 35,
            "risk_category": "MANUAL_REVIEW",
            "doc_quality_val": 70
        }

        res = self.rag_service.generate_explanation(
            document_type="NON_EXISTENT_DOC_TYPE",
            extracted_text=extracted_text,
            analysis_results=analysis_results
        )

        self.assertIsNotNone(res)
        # Should gracefully fallback to General Guidelines
        self.assertGreaterEqual(len(res["retrieved_context"]), 1)
        self.assertIn("explanation", res)
        self.assertGreater(len(res["explanation"]), 10)
        print("[OK] Test 3 (No matching knowledge fallback) passed")

    def test_4_ocr_text_missing(self):
        """4. Test RAG behavior when OCR text is completely missing/empty."""
        analysis_results = {
            "overall_risk_score": 45,
            "risk_category": "MANUAL_REVIEW",
            "doc_quality_val": 40,
            "ocr_conf_val": 0,
            "tamper_risk_val": 25
        }

        res = self.rag_service.generate_explanation(
            document_type="AADHAAR",
            extracted_text="",  # Empty OCR text
            analysis_results=analysis_results
        )

        self.assertIsNotNone(res)
        self.assertGreaterEqual(len(res["evidence_found"]), 1)
        self.assertIn("Aadhaar", res["explanation"])
        self.assertIn("manual", res["explanation"].lower())
        print("[OK] Test 4 (OCR text missing) passed")

    def test_5_unknown_document_type(self):
        """5. Test unknown document type handling."""
        res = self.rag_service.generate_explanation(
            document_type="OTHER",
            extracted_text="Unknown identification card number 987654",
            analysis_results={"overall_risk_score": 25, "risk_category": "LOW_RISK"}
        )

        self.assertEqual(res["document_type"], "OTHER")
        self.assertIsNotNone(res["explanation"])
        self.assertGreaterEqual(len(res["relevant_rules"]), 1)
        print("[OK] Test 5 (Unknown document type) passed")

    def test_6_rag_service_failure_handling(self):
        """6. Test RAG resilience when vector search encounters unexpected input or errors."""
        # Query with empty or unusual objects
        res = self.rag_service.generate_explanation(
            document_type="",
            extracted_text="",
            analysis_results={}
        )

        self.assertIsNotNone(res)
        self.assertIn("explanation", res)
        self.assertIn("risk_assessment", res)
        print("[OK] Test 6 (RAG service failure resilience) passed")


if __name__ == '__main__':
    unittest.main()
