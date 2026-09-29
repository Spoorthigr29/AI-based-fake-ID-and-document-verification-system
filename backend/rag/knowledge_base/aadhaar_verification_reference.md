# Aadhaar Card Official Verification Knowledge Reference

**Document Type:** AADHAAR
**Issuing Authority:** Unique Identification Authority of India (UIDAI), Government of India
**Statutory Framework:** Aadhaar (Targeted Delivery of Financial and Other Subsidies, Benefits and Services) Act, 2016

---

## 1. Expected Document Fields and Structure

An authentic Aadhaar document contains the following standard demographic and biometric indicators:
1. **12-Digit Aadhaar Identification Number:**
   - Formatted as `XXXX XXXX XXXX` (3 groups of 4 digits separated by spaces).
   - The number complies with the mathematical **Verhoeff Checksum Algorithm** to prevent transcription errors.
   - For privacy/security, masked Aadhaar documents display only the last 4 digits (e.g., `XXXX XXXX 1234`).
2. **Cardholder Full Name:**
   - Printed prominently in English and often in the regional language of issuance.
3. **Date of Birth / Year of Birth:**
   - Standard format: `DOB: DD/MM/YYYY` or `Year of Birth: YYYY`.
4. **Gender:**
   - Values: `Male` / `Female` / `Transgender` (or regional language equivalent).
5. **Residential Address:**
   - Full residential address with mandatory 6-digit Indian Postal PIN code.
6. **Government Insignia:**
   - National Emblem of India (Lion Capital of Ashoka) and UIDAI corporate emblem with official tagline.

---

## 2. Document Layouts and Forms

Aadhaar documents are officially issued and distributed in four genuine formats:
1. **Standard Aadhaar Letter:**
   - Two-part physical sheet. The upper section contains greeting and enrollment details; the lower section contains the detachable identity card with portrait photo, demographic text, and a prominent QR code.
2. **e-Aadhaar (Digital PDF):**
   - Digitally signed PDF with Adobe cryptographic signature verification checkmark and embedded high-density QR code.
3. **PVC Aadhaar Card:**
   - Durable wallet-sized plastic card with holographic security overlay, ghost image, micro-text, and secure QR code.
4. **mAadhaar Mobile Card:**
   - Digital profile displayed inside the official UIDAI mobile application with dynamic QR representation.

---

## 3. QR Code & Digital Signature Specifications

The QR code is the primary machine-verifiable security feature on an Aadhaar card:
1. **Secure QR Code (V2 Specification):**
   - High-density binary QR code containing gzip-compressed, base64-encoded structured demographic data.
   - Digitally signed using UIDAI's 2048-bit RSA / ECC private key.
   - Contains an embedded JPEG image of the cardholder's biometric portrait (approx. 5,000–8,000 bytes) along with Name, DOB, Gender, and Address.
2. **Standard QR Code (V1 Legacy Specification):**
   - Contains plain or XML-formatted demographic text without an embedded photograph.
3. **Offline e-KYC Verification:**
   - Verification involves decoding the QR code, verifying the cryptographic digital signature against the official UIDAI public key certificate, and comparing the decoded data against printed text.

---

## 4. Official Verification Procedures

A complete verification analysis evaluates four distinct dimensions:
1. **Format Validation:** Verify that the 12-digit number satisfies the Verhoeff algorithm.
2. **Cryptographic Validation:** Verify the digital signature of the Secure QR code.
3. **QR / OCR Cross-Validation:** Cross-check the Name, DOB, Gender, and Address extracted from the QR code against the OCR text read from the physical card. A match confirms that printed text was not altered.
4. **Biometric Face Verification:** Compare the applicant's live selfie against the photograph extracted from the Secure QR code or cropped from the document surface.

---

## 5. Common Inconsistencies & Diagnostic Guidelines

When evaluating potential anomalies, observe these critical forensic distinctions:
1. **Image Degradation vs. Tampering:**
   - Tilted angles, low camera resolution, glare on laminated cards, or WhatsApp image compression can cause OCR text extraction drops or false ELA noise.
   - Low OCR confidence or missing QR codes due to cropping MUST be treated as an image quality issue requiring manual review, NOT as conclusive proof of fraud.
2. **Face Extraction Failures:**
   - If the document photo is heavily compressed, dark, or laminated with glare, facial landmarks may not be detected. This is a biometric capture limitation, not evidence of a forged identity.
3. **Splicing and Digital Modification:**
   - Only flag high tampering risk when there is localized font mismatch, inconsistent character alignment, sharp Error Level Analysis (ELA) discontinuities around numbers, or direct contradictions between QR payload and printed text.
