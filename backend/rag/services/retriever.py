import logging
from typing import List, Dict, Any, Optional
from .vector_store import LocalVectorStore
from .text_chunker import KnowledgeChunk

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Formulates context-aware queries from document verification signals
    and retrieves the most relevant reference knowledge chunks.
    """
    def __init__(self, vector_store: LocalVectorStore):
        self.vector_store = vector_store

    def build_search_query(
        self,
        document_type: str,
        extracted_text: str = "",
        analysis_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Synthesize a structured search query incorporating document type,
        forensic anomalies, QR status, face match, and OCR signals.
        """
        analysis = analysis_results or {}
        query_parts = [f"Document Type: {document_type}"]

        # Add OCR keywords or snippet
        if extracted_text:
            clean_text = " ".join(extracted_text.split()[:50])
            query_parts.append(f"OCR Content: {clean_text}")

        # Add specific forensic signals
        qr_status = analysis.get("qr_status_display") or analysis.get("qr_status")
        if qr_status:
            query_parts.append(f"QR Code status: {qr_status}")

        tamper_risk = analysis.get("tamper_risk_val") or analysis.get("tampering_probability")
        if tamper_risk and float(tamper_risk) > 20:
            query_parts.append("tampering splicing ELA anomaly digital modification")

        face_status = analysis.get("face_match_status")
        if face_status:
            query_parts.append(f"Face verification: {face_status}")

        doc_quality = analysis.get("doc_quality_val") or analysis.get("quality_score")
        if doc_quality and float(doc_quality) < 60:
            query_parts.append("image degradation blur glare reflection low quality")

        return "\n".join(query_parts)

    def retrieve(
        self,
        document_type: str,
        extracted_text: str = "",
        analysis_results: Optional[Dict[str, Any]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant knowledge base chunks for the verification case.
        """
        query = self.build_search_query(document_type, extracted_text, analysis_results)
        raw_results = self.vector_store.search(
            query=query,
            top_k=top_k,
            doc_type_filter=document_type
        )

        retrieved = []
        for chunk, score in raw_results:
            retrieved.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_type": chunk.doc_type,
                "section_title": chunk.section_title,
                "text": chunk.text,
                "relevance_score": round(float(score), 3),
                "source": chunk.metadata.get("filename", "reference.md")
            })

        return retrieved
