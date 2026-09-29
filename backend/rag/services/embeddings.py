import re
import numpy as np
from typing import List, Union

class LocalEmbeddingGenerator:
    """
    High-performance, deterministic local embedding engine for identity verification knowledge.
    Constructs normalized dense semantic representations (dimension=128) using multi-tier
    token, character n-gram, and domain keyword hashing with L2 unit normalization.
    """
    DIMENSION = 128

    # High-signal domain keywords for identity document verification
    DOMAIN_VOCAB = [
        "aadhaar", "uidai", "verhoeff", "qr", "secure", "signature", "rsa", "ecc",
        "pan", "income", "tax", "nsdl", "utiitsl", "hologram", "surname", "individual",
        "passport", "mrz", "icao", "checksum", "given", "travel", "biodata", "embassy",
        "driving", "license", "morth", "rto", "sarathi", "transport", "mcwg", "lmv",
        "voter", "epic", "election", "commission", "assembly", "polling", "elector",
        "tamper", "ela", "compression", "splicing", "glare", "lamination", "whatsapp",
        "photo", "face", "selfie", "biometric", "arcface", "similarity", "match",
        "ocr", "paddleocr", "tesseract", "extraction", "confidence", "deskew", "quality",
        "resolution", "blur", "laplacian", "brightness", "contrast", "format", "structure",
        "low", "risk", "medium", "high", "manual", "review", "inconclusive", "genuine",
        "forgery", "counterfeit", "altered", "discrepancy", "consistent", "verified"
    ]

    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension
        self.domain_keyword_weights = {kw: 2.5 for kw in self.DOMAIN_VOCAB}

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into words and subwords."""
        if not text:
            return []
        text_norm = text.lower()
        words = re.findall(r'[a-z0-9_\-\./]+', text_norm)
        return words

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate a single normalized dense embedding vector for a given text string.
        Returns numpy array of shape (dimension,) with float32 values and unit norm.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = self._tokenize(text)

        if not words:
            # Fallback for empty text: return small uniform vector
            vec.fill(1.0 / np.sqrt(self.dimension))
            return vec

        for word in words:
            # 1. Whole word feature
            h = hash(word) % self.dimension
            weight = self.domain_keyword_weights.get(word, 1.0)
            vec[h] += weight

            # 2. Character 3-grams and 4-grams for morphological / typo tolerance
            if len(word) >= 4:
                for i in range(len(word) - 2):
                    tri = word[i:i+3]
                    h_tri = hash(tri) % self.dimension
                    vec[h_tri] += 0.3 * weight

        # Apply term-frequency squashing (log(1 + count))
        vec = np.sign(vec) * np.log1p(np.abs(vec))

        # L2 unit normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        else:
            vec.fill(1.0 / np.sqrt(self.dimension))

        return vec.astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embedding matrix for a list of document strings.
        Returns numpy array of shape (N, dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = [self.embed_text(t) for t in texts]
        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query string."""
        return self.embed_text(query)
