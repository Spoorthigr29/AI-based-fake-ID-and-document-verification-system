# Institutional Identity Document Screening & Risk Assessment Guidelines

**Document Type:** GENERAL / MULTI-DOCUMENT
**Purpose:** Principles of Responsible Forensic Analysis & Explainable AI Verification

---

## 1. Core Principles of Evidence-Based Verification

1. **Multi-Signal Convergence:**
   - No single technical signal (e.g., OCR confidence drop, missing QR code, or biometric threshold variance) is independently sufficient to brand a document as counterfeit or fraudulent.
   - Genuine fraud requires converging positive evidence: deliberate font alteration, content splicing, conflicting checksums, or direct contradiction with cryptographically signed payloads.
2. **Distinguishing Acquisition Artifacts from Tampering:**
   - **Physical & Acquisition Artifacts:** Camera shake, poor ambient lighting, flash glare, lamination scratches, perspective distortion, and lossy compression (e.g., WhatsApp downsampling) degrade algorithmic metrics but do NOT indicate forgery.
   - **Forensic Tampering:** Sharp ELA compression boundaries, copy-paste cloning, mismatched background patterns, unnatural font weights, and character misalignments.
3. **Template-Aware Biometrics:**
   - Identity documents without photographs (e.g. non-photo PAN templates, corporate tax cards) must not be penalized for absent facial biometrics.
4. **Proportional Risk Scoring & Manual Review:**
   - **Low Risk (0–30):** Document conforms to expected structure, format, and biometric rules.
   - **Manual Review (31–70):** Ambiguity due to image quality, unreadable QR, or borderline biometric match. Manual officer inspection recommended.
   - **High Risk (71–100):** Converging forensic anomalies, direct cryptographic mismatch, or explicit digital tampering.
