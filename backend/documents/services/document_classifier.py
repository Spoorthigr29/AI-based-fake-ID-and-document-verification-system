import re
import os
import pypdf
import io
from typing import Dict, Any, List

class DocumentClassifier:
    """
    Multi-Signal Document Classification Engine for VerifyX AI.
    Classifies documents into standard institutional categories:
    - AADHAAR (Aadhaar Identity Document)
    - PAN (Permanent Account Number Card)
    - PASSPORT (International Passport / MRZ)
    - DRIVING_LICENSE (Motor Vehicle Driving License)
    - VOTER_ID (Election Commission Voter Identity Card / EPIC)
    - OTHER (Unclassified / Generic Document)
    """

    CLASS_AADHAAR = "AADHAAR"
    CLASS_PAN = "PAN"
    CLASS_PASSPORT = "PASSPORT"
    CLASS_DRIVING_LICENSE = "DRIVING_LICENSE"
    CLASS_VOTER_ID = "VOTER_ID"
    CLASS_OTHER = "OTHER"

    CLASS_DISPLAY_NAMES = {
        CLASS_AADHAAR: "Aadhaar Card",
        CLASS_PAN: "PAN Card",
        CLASS_PASSPORT: "Passport",
        CLASS_DRIVING_LICENSE: "Driving License",
        CLASS_VOTER_ID: "Voter ID (EPIC)",
        CLASS_OTHER: "Other Document"
    }

    # Regex signatures for national identity formats
    RE_PAN = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')
    RE_AADHAAR = re.compile(r'\b(\d{4}\s\d{4}\s\d{4}|\d{12})\b')
    RE_PASSPORT = re.compile(r'\b[A-Z][0-9]{7}\b')
    RE_DL = re.compile(r'\b([A-Z]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{11}|[A-Z]{2}[0-9]{13})\b')
    RE_VOTER = re.compile(r'\b[A-Z]{3}[0-9]{7}\b')

    @classmethod
    def extract_text_snippet(cls, file_or_path):
        """Extract available text from PDF, OCR cache, or ASCII data stream."""
        text = ""
        is_pdf = False
        if isinstance(file_or_path, str):
            is_pdf = file_or_path.lower().endswith('.pdf')
        elif hasattr(file_or_path, 'name'):
            is_pdf = (file_or_path.name or '').lower().endswith('.pdf')

        if is_pdf:
            try:
                if isinstance(file_or_path, str):
                    reader = pypdf.PdfReader(file_or_path)
                else:
                    reader = pypdf.PdfReader(io.BytesIO(file_or_path.read() if hasattr(file_or_path, 'read') else file_or_path))
                for p in reader.pages[:2]:
                    t = p.extract_text()
                    if t:
                        text += " " + t
            except Exception:
                pass

        if not text and isinstance(file_or_path, (str, os.PathLike)) and os.path.exists(file_or_path):
            try:
                with open(file_or_path, 'rb') as f:
                    raw_b = f.read(10000)
                    strings = re.findall(rb'[A-Za-z0-9\s,:\-/\.]{4,100}', raw_b)
                    for s in strings:
                        text += " " + s.decode('utf-8', errors='ignore')
            except Exception:
                pass
        return text

    @classmethod
    def classify(cls, file_or_path, ocr_text=None) -> Dict[str, Any]:
        """
        Classify document and return:
        - document_type: str ('AADHAAR', 'PAN', 'PASSPORT', 'DRIVING_LICENSE', 'VOTER_ID', 'OTHER')
        - display_name: str
        - confidence: float (0.0 to 1.0)
        - probabilities: dict of top class probabilities
        - signals: list of detected marker strings
        """
        combined_text = (ocr_text or "") + " " + cls.extract_text_snippet(file_or_path)
        text_upper = combined_text.upper()

        scores = {
            cls.CLASS_AADHAAR: 0.05,
            cls.CLASS_PAN: 0.05,
            cls.CLASS_PASSPORT: 0.05,
            cls.CLASS_DRIVING_LICENSE: 0.05,
            cls.CLASS_VOTER_ID: 0.05,
            cls.CLASS_OTHER: 0.10
        }
        signals = []

        # 1. Aadhaar Identity Signatures
        aadhaar_keywords = ["UNIQUE IDENTIFICATION", "AADHAAR", "UIDAI", "AUTHORITY OF INDIA", "MERA AADHAAR", "GOVERNMENT OF INDIA", "ENROLMENT NO", "VID:"]
        aadhaar_hits = [k for k in aadhaar_keywords if k in text_upper]
        if aadhaar_hits:
            scores[cls.CLASS_AADHAAR] += 0.40 * len(aadhaar_hits)
            signals.append(f"Aadhaar / UIDAI Header Keywords: {', '.join(aadhaar_hits[:3])}")

        if cls.RE_AADHAAR.search(combined_text):
            scores[cls.CLASS_AADHAAR] += 0.35
            signals.append("12-Digit Identity Number Format")

        if any(k in text_upper for k in ["DOB", "YEAR OF BIRTH", "MALE", "FEMALE", "ADDRESS"]) and ("INDIA" in text_upper or "GOVERNMENT" in text_upper):
            scores[cls.CLASS_AADHAAR] += 0.20

        # 2. PAN Card Signatures
        pan_keywords = ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "INCOMETAX", "GOVT. OF INDIA", "FATHER'S NAME", "SIGNATURE"]
        pan_hits = [k for k in pan_keywords if k in text_upper]
        if pan_hits:
            scores[cls.CLASS_PAN] += 0.45 * len(pan_hits)
            signals.append(f"Income Tax / PAN Keywords: {', '.join(pan_hits[:3])}")

        if cls.RE_PAN.search(text_upper):
            scores[cls.CLASS_PAN] += 0.45
            signals.append("10-Character Alphanumeric PAN Format")

        # 3. Passport Signatures
        passport_keywords = ["PASSPORT", "REPUBLIC OF INDIA", "P<IND", "TYPE P", "CODE IND", "PASSPORT NO", "NATIONALITY"]
        pass_hits = [k for k in passport_keywords if k in text_upper]
        if pass_hits:
            scores[cls.CLASS_PASSPORT] += 0.45 * len(pass_hits)
            signals.append(f"Passport / MRZ Keywords: {', '.join(pass_hits[:3])}")

        if cls.RE_PASSPORT.search(text_upper) and any(k in text_upper for k in ["REPUBLIC", "VISA", "EXPIRY", "SURNAME"]):
            scores[cls.CLASS_PASSPORT] += 0.35
            signals.append("Passport Number & Attribute Signature")

        # 4. Driving License Signatures
        dl_keywords = ["DRIVING LICENCE", "DRIVING LICENSE", "UNION OF INDIA DRIVING", "DL NO", "TRANSPORT DEPARTMENT", "FORM 7", "MOTOR VEHICLES", "AUTHORISATION TO DRIVE"]
        dl_hits = [k for k in dl_keywords if k in text_upper]
        if dl_hits:
            scores[cls.CLASS_DRIVING_LICENSE] += 0.45 * len(dl_hits)
            signals.append(f"Driving License Keywords: {', '.join(dl_hits[:3])}")

        if cls.RE_DL.search(text_upper):
            scores[cls.CLASS_DRIVING_LICENSE] += 0.40
            signals.append("Driving License Number Format")

        # 5. Voter ID Signatures
        voter_keywords = ["ELECTION COMMISSION", "ELECTORAL PHOTO", "EPIC", "VOTER IDENTITY", "BHARAT NIRVACHAN"]
        voter_hits = [k for k in voter_keywords if k in text_upper]
        if voter_hits:
            scores[cls.CLASS_VOTER_ID] += 0.50 * len(voter_hits)
            signals.append(f"Voter ID / EPIC Keywords: {', '.join(voter_hits[:2])}")

        if cls.RE_VOTER.search(text_upper):
            scores[cls.CLASS_VOTER_ID] += 0.35

        # Normalize score vector into probability distribution (Softmax-like scaling)
        max_raw = max(scores.values())
        if max_raw > 0.15:
            scores[cls.CLASS_OTHER] = 0.05
        else:
            scores[cls.CLASS_OTHER] = 0.60

        total = sum(scores.values())
        probs = {k: round(v / total, 4) for k, v in scores.items()}
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        top_class, top_prob = sorted_probs[0]
        # Calibrate confidence
        if top_class != cls.CLASS_OTHER and top_prob >= 0.35:
            confidence = min(0.98, max(0.70, top_prob + 0.25))
        elif top_class != cls.CLASS_OTHER:
            confidence = round(top_prob, 2)
        else:
            confidence = 0.50
            top_class = cls.CLASS_OTHER

        top_3 = {k: round(v, 4) for k, v in sorted_probs[:3]}

        return {
            "document_type": top_class,
            "display_name": cls.CLASS_DISPLAY_NAMES.get(top_class, "Unknown Document"),
            "confidence": round(confidence, 2),
            "probabilities": top_3,
            "signals": signals if signals else ["General visual document layout"]
        }

