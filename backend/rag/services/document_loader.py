import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class KnowledgeDocument:
    doc_id: str
    doc_type: str
    title: str
    content: str
    file_path: str
    metadata: Dict[str, Any]

class DocumentLoader:
    """
    Loads official reference knowledge documents from markdown or text files
    in the knowledge_base directory.
    """
    def __init__(self, kb_dir: Optional[str] = None):
        if kb_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.kb_dir = os.path.join(base_dir, 'knowledge_base')
        else:
            self.kb_dir = kb_dir

    def load_documents(self) -> List[KnowledgeDocument]:
        """Load all markdown and text documents from knowledge_base directory."""
        if not os.path.exists(self.kb_dir):
            return []

        doc_files = glob.glob(os.path.join(self.kb_dir, '*.md')) + glob.glob(os.path.join(self.kb_dir, '*.txt'))
        documents = []

        for file_path in doc_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                filename = os.path.basename(file_path)
                doc_id = os.path.splitext(filename)[0]

                # Extract title and document type
                title = doc_id.replace('_', ' ').title()
                doc_type = 'GENERAL'

                for line in content.splitlines()[:15]:
                    line_clean = line.strip()
                    if line_clean.startswith('# '):
                        title = line_clean.replace('# ', '').strip()
                    if '**Document Type:**' in line_clean:
                        doc_type = line_clean.split('**Document Type:**')[1].strip().upper()
                    elif 'Document Type:' in line_clean:
                        doc_type = line_clean.split('Document Type:')[1].strip().upper()

                doc = KnowledgeDocument(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    title=title,
                    content=content,
                    file_path=file_path,
                    metadata={
                        "filename": filename,
                        "doc_type": doc_type,
                        "title": title
                    }
                )
                documents.append(doc)
            except Exception as e:
                continue

        return documents
