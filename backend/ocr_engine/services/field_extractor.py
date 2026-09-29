import re
from datetime import datetime

class FieldExtractor:
    """
    Intelligent field extraction engine tailored for synthetic and official identity documents:
    - Aadhaar Card (UID)
    - Permanent Account Number (PAN) Card
    - Passport (including MRZ parsing)
    - Driving License
    - Voter ID (EPIC)
    - Generic Government ID

    Extracts: name, date_of_birth, document_number, address, gender, issue_date, expiry_date, document_type.
    Does NOT assume all fields are present in every document.
    """

    # Compiled Regex Patterns
    RE_PAN = re.compile(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b')
    RE_AADHAAR = re.compile(r'\b(\d{4}\s\d{4}\s\d{4})\b')
    RE_AADHAAR_RAW = re.compile(r'\b(\d{12})\b')
    RE_PASSPORT = re.compile(r'\b([A-Z][0-9]{7})\b')
    RE_PASSPORT_PREFIX = re.compile(r'(?:Passport No|Passport Number|Passport\s*#)\s*[:\-]?\s*([A-Z][0-9]{7})\b', re.IGNORECASE)
    RE_VOTER_ID = re.compile(r'\b([A-Z]{3}[0-9]{7})\b')
    RE_DL = re.compile(r'\b([A-Z]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{11}|[A-Z]{2}[0-9]{13})\b')
    
    RE_DOB = re.compile(
        r'(?:DOB|Date of Birth|D\.O\.B|Birth|Born|DOB\s*:)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})',
        re.IGNORECASE
    )
    RE_YEAR_ONLY = re.compile(r'(?:Year of Birth|YOB)\s*[:\-]?\s*(\d{4})', re.IGNORECASE)
    RE_DATE_GENERIC = re.compile(r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b')
    RE_EXPIRY = re.compile(r'(?:Expiry\s*Date|Date\s*of\s*Expiry|Valid\s*Till|Valid\s*Upto|Expires|Exp\s*Date|Expiry)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})', re.IGNORECASE)
    RE_ISSUE = re.compile(r'(?:Issue\s*Date|Issued\s*On|Date\s*of\s*Issue|DOI|Issued)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})', re.IGNORECASE)
    
    RE_GENDER = re.compile(r'\b(FEMALE|MALE|TRANSGENDER|FEM|TRANS|M|F)\b', re.IGNORECASE)
    RE_PINCODE = re.compile(r'\b([1-9][0-9]{5})\b')

    @classmethod
    def detect_document_type(cls, raw_text):
        """
        Classify document template format based on textual marker signatures.
        """
        text_upper = raw_text.upper()
        if "INCOME TAX" in text_upper or "PERMANENT ACCOUNT NUMBER" in text_upper or "PAN CARD" in text_upper:
            return "PAN", 0.95
        elif "UNIQUE IDENTIFICATION" in text_upper or "AADHAAR" in text_upper or "UIDAI" in text_upper or ("GOVERNMENT OF INDIA" in text_upper and "ENROLMENT" in text_upper):
            return "AADHAAR", 0.94
        elif "PASSPORT" in text_upper or ("REPUBLIC OF INDIA" in text_upper and "P<IND" in text_upper):
            return "PASSPORT", 0.96
        elif "DRIVING LICENCE" in text_upper or "DRIVING LICENSE" in text_upper or "UNION OF INDIA DRIVING" in text_upper or "DL NO" in text_upper:
            return "DRIVING_LICENSE", 0.92
        elif "ELECTION COMMISSION" in text_upper or "ELECTOR PHOTO" in text_upper or "EPIC" in text_upper:
            return "VOTER_ID", 0.92
        elif cls.RE_PAN.search(text_upper):
            return "PAN", 0.88
        elif cls.RE_AADHAAR.search(raw_text) or cls.RE_AADHAAR_RAW.search(raw_text):
            return "AADHAAR", 0.85
        elif cls.RE_PASSPORT_PREFIX.search(text_upper) or cls.RE_PASSPORT.search(text_upper):
            return "PASSPORT", 0.80
        elif cls.RE_VOTER_ID.search(text_upper):
            return "VOTER_ID", 0.80
        return "OTHER", 0.60

    @classmethod
    def standardize_date(cls, date_str):
        """Standardize date strings to YYYY-MM-DD format."""
        if not date_str:
            return None
        date_clean = date_str.strip().replace('/', '-')
        for fmt in ('%d-%m-%Y', '%d-%m-%y', '%Y-%m-%d', '%m-%d-%Y', '%Y'):
            try:
                dt = datetime.strptime(date_clean, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return date_str

    @classmethod
    def extract_fields(cls, ocr_results, base_confidence=0.85):
        """
        Extract structured fields from OCR output.
        """
        if isinstance(ocr_results, list):
            lines = []
            confidences = []
            for item in ocr_results:
                if isinstance(item, dict):
                    t = item.get('text', '').strip()
                    c = item.get('confidence', base_confidence)
                elif isinstance(item, tuple) and len(item) == 2:
                    t = str(item[0]).strip()
                    c = float(item[1])
                else:
                    t = str(item).strip()
                    c = base_confidence
                if t:
                    lines.append(t)
                    confidences.append(c)
            raw_text = "\n".join(lines)
            avg_ocr_conf = float(sum(confidences) / max(1, len(confidences))) if confidences else base_confidence
        else:
            raw_text = str(ocr_results or '')
            lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
            avg_ocr_conf = base_confidence

        extracted = {}

        if not raw_text.strip():
            return {
                "fields": {},
                "raw_text": "",
                "detected_type": "UNKNOWN",
                "overall_confidence": 0.0
            }

        # 1. Document Type Detection
        doc_type, type_conf = cls.detect_document_type(raw_text)
        extracted["document_type"] = {
            "value": doc_type,
            "confidence": round(min(1.0, (avg_ocr_conf + type_conf) / 2), 2)
        }

        # 2. Document Number Extraction
        doc_num = None
        doc_num_conf = 0.0

        if doc_type == "PAN":
            m = cls.RE_PAN.search(raw_text.upper())
            if m:
                doc_num = m.group(1)
                doc_num_conf = 0.98
        elif doc_type == "AADHAAR":
            m = cls.RE_AADHAAR.search(raw_text) or cls.RE_AADHAAR_RAW.search(raw_text)
            if m:
                val = m.group(1).replace(" ", "")
                doc_num = f"{val[0:4]} {val[4:8]} {val[8:12]}"
                doc_num_conf = 0.96
        elif doc_type == "PASSPORT":
            m_prefix = cls.RE_PASSPORT_PREFIX.search(raw_text)
            if m_prefix:
                doc_num = m_prefix.group(1).upper()
                doc_num_conf = 0.97
            else:
                m = cls.RE_PASSPORT.search(raw_text.upper())
                if m:
                    doc_num = m.group(1).upper()
                    doc_num_conf = 0.95
        elif doc_type == "VOTER_ID":
            m = cls.RE_VOTER_ID.search(raw_text.upper())
            if m:
                doc_num = m.group(1).upper()
                doc_num_conf = 0.95
        elif doc_type == "DRIVING_LICENSE":
            m = cls.RE_DL.search(raw_text.upper())
            if m:
                doc_num = m.group(1).upper()
                doc_num_conf = 0.94

        # Fallback number search
        if not doc_num:
            for pat, conf in [(cls.RE_PAN, 0.90), (cls.RE_AADHAAR, 0.90), (cls.RE_PASSPORT_PREFIX, 0.90), (cls.RE_PASSPORT, 0.88), (cls.RE_VOTER_ID, 0.88), (cls.RE_DL, 0.85)]:
                m = pat.search(raw_text)
                if m:
                    doc_num = m.group(1)
                    doc_num_conf = conf
                    break

        if doc_num:
            extracted["document_number"] = {
                "value": doc_num,
                "confidence": round(min(1.0, doc_num_conf * (0.5 + 0.5 * avg_ocr_conf)), 2)
            }

        # 3. Date of Birth Extraction
        dob = None
        dob_match = cls.RE_DOB.search(raw_text)
        if dob_match:
            dob = cls.standardize_date(dob_match.group(1))
            dob_conf = 0.96
        else:
            yob_match = cls.RE_YEAR_ONLY.search(raw_text)
            if yob_match:
                dob = f"{yob_match.group(1)}-01-01"
                dob_conf = 0.85
            else:
                dates = cls.RE_DATE_GENERIC.findall(raw_text)
                if dates:
                    dob = cls.standardize_date(dates[0])
                    dob_conf = 0.80

        if dob:
            extracted["date_of_birth"] = {
                "value": dob,
                "confidence": round(min(1.0, dob_conf * (0.5 + 0.5 * avg_ocr_conf)), 2)
            }

        # 4. Gender Extraction
        gender_match = cls.RE_GENDER.search(raw_text)
        if gender_match:
            g_raw = gender_match.group(1).upper()
            if g_raw in ["FEMALE", "FEM", "F"]:
                gender_val = "FEMALE"
            elif g_raw in ["TRANSGENDER", "TRANS"]:
                gender_val = "TRANSGENDER"
            else:
                gender_val = "MALE"
            extracted["gender"] = {
                "value": gender_val,
                "confidence": round(min(1.0, 0.95 * avg_ocr_conf), 2)
            }

        # 5. Issue Date & Expiry Date
        issue_match = cls.RE_ISSUE.search(raw_text)
        if issue_match:
            extracted["issue_date"] = {
                "value": cls.standardize_date(issue_match.group(1)),
                "confidence": round(min(1.0, 0.92 * avg_ocr_conf), 2)
            }

        expiry_match = cls.RE_EXPIRY.search(raw_text)
        if expiry_match:
            extracted["expiry_date"] = {
                "value": cls.standardize_date(expiry_match.group(1)),
                "confidence": round(min(1.0, 0.92 * avg_ocr_conf), 2)
            }

        # 6. Name Extraction (Smart clean heuristics)
        name_val = None
        name_conf = 0.0

        blacklisted_phrases = [
            "GOVERNMENT OF INDIA", "INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER",
            "REPUBLIC OF INDIA", "ELECTION COMMISSION", "DRIVING LICENCE", "UNION OF INDIA",
            "MALE", "FEMALE", "FATHER'S NAME", "SIGNATURE", "DATE OF BIRTH", "DOB",
            "ADDRESS", "AUTHORITY", "AADHAAR", "CARD", "PASSPORT", "INDIA", "SURNAME", "GIVEN NAME"
        ]

        name_prefix_match = re.search(r'(?:Name|Name of Holder|Full Name|Given Name)\s*[:\-]?\s*([A-Za-z\s\.]{3,40})', raw_text, re.IGNORECASE)
        if name_prefix_match:
            candidate = " ".join(name_prefix_match.group(1).strip().split())
            if len(candidate) > 2 and not any(b in candidate.upper() for b in blacklisted_phrases):
                name_val = candidate.title()
                name_conf = 0.96

        if not name_val:
            for line in lines:
                clean_l = re.sub(r'[^A-Za-z\s]', '', line).strip()
                words = clean_l.split()
                if 2 <= len(words) <= 4 and len(clean_l) >= 4:
                    upper_l = clean_l.upper()
                    if not any(b in upper_l for b in blacklisted_phrases):
                        name_val = " ".join(words).title()
                        name_conf = 0.88
                        break

        if name_val:
            extracted["name"] = {
                "value": name_val,
                "confidence": round(min(1.0, name_conf * (0.5 + 0.5 * avg_ocr_conf)), 2)
            }

        # 7. Address Extraction
        address_lines = []
        for line in lines:
            if cls.RE_PINCODE.search(line) or any(k in line.upper() for k in ["ROAD", "STREET", "NAGAR", "COLONY", "DIST", "DISTRICT", "STATE", "PIN", "PO", "HOUSE", "FLAT", "BANGALORE", "MUMBAI", "DELHI"]):
                if not any(b in line.upper() for b in ["INCOME TAX", "GOVERNMENT", "ELECTION", "PASSPORT", "REPUBLIC"]):
                    address_lines.append(" ".join(line.strip().split()))

        if address_lines:
            extracted["address"] = {
                "value": ", ".join(address_lines[:3]),
                "confidence": round(min(1.0, 0.86 * avg_ocr_conf), 2)
            }

        all_confs = [f["confidence"] for f in extracted.values()]
        overall_conf = round(float(sum(all_confs) / max(1, len(all_confs))), 2)

        return {
            "fields": extracted,
            "raw_text": raw_text,
            "detected_type": doc_type,
            "overall_confidence": overall_conf
        }
