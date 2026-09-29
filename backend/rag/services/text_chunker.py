import re
from dataclasses import dataclass
from typing import List, Dict, Any
from .document_loader import KnowledgeDocument

@dataclass
class KnowledgeChunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    section_title: str
    text: str
    metadata: Dict[str, Any]

class TextChunker:
    """
    Splits knowledge documents into structured, semantically meaningful chunks
    based on markdown section headers and paragraphs with overlap.
    """
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: KnowledgeDocument) -> List[KnowledgeChunk]:
        """Split a single knowledge document into semantic chunks."""
        chunks: List[KnowledgeChunk] = []
        raw_content = doc.content

        # Split by markdown headers (## or ###)
        sections = re.split(r'\n(?=##+\s+)', raw_content)
        chunk_seq = 0

        for section in sections:
            section_clean = section.strip()
            if not section_clean:
                continue

            # Extract section title if present
            lines = section_clean.splitlines()
            first_line = lines[0].strip()
            if first_line.startswith('#'):
                section_title = re.sub(r'^#+\s*', '', first_line).strip()
            else:
                section_title = doc.title

            # If section text is compact enough, keep as single chunk
            words = section_clean.split()
            if len(words) <= self.chunk_size:
                chunk_seq += 1
                chunk_id = f"{doc.doc_id}_chunk_{chunk_seq}"
                chunks.append(KnowledgeChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    doc_type=doc.doc_type,
                    section_title=section_title,
                    text=section_clean,
                    metadata={
                        **doc.metadata,
                        "chunk_id": chunk_id,
                        "section_title": section_title,
                        "word_count": len(words)
                    }
                ))
            else:
                # Sub-chunk longer sections with sliding window
                step = self.chunk_size - self.chunk_overlap
                for i in range(0, len(words), step):
                    sub_words = words[i:i + self.chunk_size]
                    if not sub_words:
                        continue
                    chunk_seq += 1
                    chunk_id = f"{doc.doc_id}_chunk_{chunk_seq}"
                    chunk_text = f"## {section_title}\n" + " ".join(sub_words)
                    chunks.append(KnowledgeChunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        doc_type=doc.doc_type,
                        section_title=section_title,
                        text=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_id": chunk_id,
                            "section_title": section_title,
                            "word_count": len(sub_words)
                        }
                    ))

        return chunks

    def chunk_documents(self, docs: List[KnowledgeDocument]) -> List[KnowledgeChunk]:
        """Chunk a collection of knowledge documents."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
