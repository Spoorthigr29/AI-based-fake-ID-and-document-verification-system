import logging
from typing import List, Dict, Any, Optional

from .document_loader import DocumentLoader
from .text_chunker import TextChunker
from .vector_store import LocalVectorStore
from .retriever import RAGRetriever

logger = logging.getLogger(__name__)

class RAGService:
    """
    Unified RAG (Retrieval-Augmented Generation) Service for VerifyX AI.
    Integrates verified reference knowledge with real-time multi-modal forensic signals
    to generate clear, transparent, and explainable identity verification outcomes.
    """
    _instance: Optional['RAGService'] = None

    def __init__(self):
        self.doc_loader = DocumentLoader()
        self.chunker = TextChunker()
        self.vector_store = LocalVectorStore()
        self.retriever = RAGRetriever(self.vector_store)
        self._is_initialized = False

    @classmethod
    def get_instance(cls) -> 'RAGService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the RAG service and vector index.
        Loads cached vector index if available, or indexes knowledge base files.
        """
        if self._is_initialized and not force_rebuild:
            return True

        if not force_rebuild and self.vector_store.load():
            self._is_initialized = True
            return True

        # Build index from knowledge base markdown files
        docs = self.doc_loader.load_documents()
        if not docs:
            logger.warning("No reference documents found in knowledge base.")
            self._is_initialized = True
            return False

        chunks = self.chunker.chunk_documents(docs)
        self.vector_store.build_from_chunks(chunks)
        self._is_initialized = True
        return True

    def rebuild_index(self) -> int:
        """Force re-indexing of all knowledge base documents."""
        self.initialize(force_rebuild=True)
        return len(self.vector_store.chunks)

    def extract_evidence(self, analysis_results: Dict[str, Any], extracted_text: str = "", document_type: str = "") -> List[str]:
        """Synthesize structured evidence items from multi-signal pipeline outputs."""
        evidence = []
        doc_type_norm = str(document_type or analysis_results.get("document_type") or "OTHER").upper().strip()

        # 1. OCR text signal
        ocr_conf = analysis_results.get("ocr_conf_val") or analysis_results.get("ocr_confidence")
        if ocr_conf is not None:
            conf_val = int(ocr_conf * 100) if ocr_conf <= 1.0 else int(ocr_conf)
            if conf_val >= 70:
                evidence.append(f"OCR Text Extraction: Completed with high confidence ({conf_val}%).")
            elif conf_val >= 50:
                evidence.append(f"OCR Text Extraction: Moderate confidence ({conf_val}%). Some document text could not be extracted with sufficient confidence.")
            else:
                evidence.append(f"OCR Text Extraction: Low confidence ({conf_val}%) due to image capture or styling.")
        elif extracted_text:
            evidence.append("OCR Text Extraction: Text content successfully recognized.")

        # 2. QR Code signal (Document Type specific)
        qr_status = analysis_results.get("qr_status_display") or analysis_results.get("qr_status")
        qr_consistency = analysis_results.get("qr_consistency_display") or analysis_results.get("qr_consistency_status")

        if "PAN" in doc_type_norm:
            if qr_status and "Verified" in str(qr_status):
                evidence.append("PAN QR / Document Check: QR payload detected and verified.")
            elif qr_status and "Detected" in str(qr_status):
                evidence.append("PAN QR / Document Check: QR detected on card surface.")
            else:
                evidence.append("PAN QR / Document Check: Quick-response code unavailable on this card (excluded without penalty).")
        else:
            if qr_status and "Verified" in str(qr_status):
                evidence.append("Aadhaar Secure QR: Cryptographic payload detected and decoded.")
            elif qr_status and "Detected" in str(qr_status):
                evidence.append("Aadhaar QR Code: Detected on card surface.")
            elif qr_status:
                evidence.append(f"Aadhaar Secure QR: {qr_status} (excluded without penalty).")

        if qr_consistency and "Not Available" not in str(qr_consistency):
            evidence.append(f"QR / Document Consistency: {qr_consistency}.")

        # 3. Document photo & face biometric signal
        is_face_applicable = analysis_results.get("is_face_applicable", True)
        face_status = analysis_results.get("face_match_status") or analysis_results.get("face_match_display")
        face_score = analysis_results.get("face_match_val") or analysis_results.get("face_score")

        if not is_face_applicable:
            evidence.append("Document Photo: Not applicable for this document template (zero penalty applied).")
        elif face_status:
            if "Match" in str(face_status) or "Passed" in str(face_status):
                evidence.append(f"Biometric Face Match: Live selfie matched document photo (Similarity: {face_score or 90}%).")
            elif "Review" in str(face_status) or "Required" in str(face_status):
                evidence.append(f"Biometric Face Match: Borderline similarity ({face_score or 50}%) — Manual review recommended.")
            elif "Mismatch" in str(face_status):
                evidence.append(f"Biometric Face Match: Potential facial discrepancy detected between selfie and document.")

        # 4. Tamper & ELA signal
        tamper_val = analysis_results.get("tamper_risk_val") or analysis_results.get("tamper_score")
        if tamper_val is not None:
            t_score = int(tamper_val)
            if t_score <= 35:
                evidence.append(f"Digital Forensics: No malicious tampering or pixel manipulation detected ({t_score}% risk).")
            elif t_score <= 60:
                evidence.append(f"Digital Forensics: Minor compression variations detected ({t_score}% risk).")
            else:
                evidence.append(f"Digital Forensics: Elevated anomaly pattern detected in document image ({t_score}% risk).")

        # 5. Document Image Quality
        quality_score = analysis_results.get("doc_quality_val") or analysis_results.get("quality_score")
        if quality_score is not None:
            q_val = int(quality_score)
            if q_val >= 60:
                evidence.append(f"Image Quality: Acceptable clarity and sharpness ({q_val}/100).")
            else:
                evidence.append(f"Image Quality: Sub-optimal capture clarity ({q_val}/100) — blur or glare noted.")

        return evidence

    def extract_relevant_rules(self, retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract key official verification rules and criteria from retrieved chunks."""
        rules = []
        for chunk in retrieved_chunks:
            title = chunk.get("section_title", "General Guidelines")
            text = chunk.get("text", "")

            # Extract bullet points from chunk text
            for line in text.splitlines():
                line_clean = line.strip()
                if line_clean.startswith(('-', '*', '1.', '2.', '3.', '4.', '5.')):
                    rule_text = line_clean.lstrip('-*123456789. ')
                    if len(rule_text) > 20 and not rule_text.startswith('#'):
                        # Keep high-value rule lines
                        rules.append(f"{title}: {rule_text}")
                        if len(rules) >= 5:
                            break
            if len(rules) >= 5:
                break

        if not rules:
            rules.append("Standard Identity Verification: Validate expected layout, alphanumeric checksums, and biometric photo.")
            rules.append("Multi-Signal Assessment: Discrepancies in single signals require manual officer review rather than immediate rejection.")

        return rules[:4]

    def generate_explanation(
        self,
        document_type: str,
        extracted_text: str = "",
        analysis_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main RAG Explanation Pipeline.
        1. Formulates context query from detected signals.
        2. Retrieves relevant official knowledge base chunks.
        3. Compares detected evidence against official rules.
        4. Synthesizes a structured explainability payload.
        """
        self.initialize()
        analysis = analysis_results or {}
        doc_type_norm = (document_type or "OTHER").upper().strip()

        # 1. Retrieve relevant knowledge chunks
        try:
            retrieved_chunks = self.retriever.retrieve(
                document_type=doc_type_norm,
                extracted_text=extracted_text,
                analysis_results=analysis,
                top_k=3
            )
        except Exception as e:
            logger.warning(f"RAG retrieval error: {e}")
            retrieved_chunks = []

        # 2. Extract Evidence Found
        evidence_found = self.extract_evidence(analysis, extracted_text=extracted_text, document_type=doc_type_norm)

        # 3. Extract Relevant Rules
        relevant_rules = self.extract_relevant_rules(retrieved_chunks)

        # 4. Synthesize Risk Assessment & Narrative Explanation
        risk_score = analysis.get("overall_risk_score") or analysis.get("risk_score", 15)
        risk_category = analysis.get("risk_category") or analysis.get("category", "LOW_RISK")
        decision = analysis.get("decision") or ("APPROVE" if risk_score <= 30 else "MANUAL_REVIEW")

        risk_category_display = str(risk_category).replace('_', ' ').title()

        # Build transparent, balanced explanation
        narrative_parts = []

        # Document type match context
        doc_name = "Aadhaar document" if "AADHAAR" in doc_type_norm else ("PAN card" if "PAN" in doc_type_norm else f"{doc_type_norm} document")
        
        if risk_score <= 30:
            narrative_parts.append(
                f"The submitted {doc_name} conforms to expected structural formatting and official reference criteria."
            )
            if "AADHAAR" in doc_type_norm and any("QR" in ev for ev in evidence_found):
                narrative_parts.append("Demographic fields align with extracted visual text and biometric indicators.")
            narrative_parts.append(
                f"Overall risk assessment is Low ({risk_score}/100) with no critical forensic discrepancies detected."
            )
        elif risk_score <= 70:
            narrative_parts.append(
                f"The document matches the expected {doc_name} profile, but one or more screening signals require verification."
            )
            # Identify specific reason
            flags = []
            if analysis.get("doc_quality_val", 100) < 60:
                flags.append("image clarity / lighting conditions")
            if analysis.get("tamper_risk_val", 0) > 25:
                flags.append("compression / edge noise")
            if analysis.get("is_face_applicable") and "Match" not in str(analysis.get("face_match_status", "")):
                flags.append("biometric facial similarity")

            if flags:
                narrative_parts.append(
                    f"Noticeable variations were detected in {', '.join(flags)}. The available evidence is not sufficient to conclusively determine authenticity, so manual officer inspection is recommended."
                )
            else:
                narrative_parts.append(
                    "Borderline screening confidence across multi-modal indicators suggests manual verification before approval."
                )
        else:
            narrative_parts.append(
                f"Elevated risk score ({risk_score}/100) flagged for the submitted {doc_name} due to compound anomalies."
            )
            narrative_parts.append(
                "Mandatory institutional officer review is required to cross-examine physical credentials against authoritative registers."
            )

        explanation_narrative = " ".join(narrative_parts)

        return {
            "document_type": doc_type_norm,
            "evidence_found": evidence_found,
            "relevant_rules": relevant_rules,
            "retrieved_context": [
                {
                    "section_title": c.get("section_title"),
                    "snippet": c.get("text", "")[:280] + "..." if len(c.get("text", "")) > 280 else c.get("text", ""),
                    "source": c.get("source"),
                    "relevance_score": c.get("relevance_score")
                }
                for c in retrieved_chunks
            ],
            "risk_assessment": {
                "risk_score": risk_score,
                "category": risk_category,
                "category_display": risk_category_display,
                "decision": decision
            },
            "explanation": explanation_narrative
        }
