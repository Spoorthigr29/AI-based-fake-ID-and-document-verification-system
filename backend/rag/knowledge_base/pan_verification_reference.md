# Permanent Account Number (PAN) Card Official Verification Reference

**Document Type:** PAN
**Issuing Authority:** Income Tax Department, Ministry of Finance, Government of India (managed through Protean / NSDL e-Gov and UTIITSL)
**Statutory Framework:** Income Tax Act, 1961 (Section 139A)

---

## 1. Expected Document Fields and Structure

An authentic PAN card contains the following standard demographic and institutional fields:
1. **Permanent Account Number (PAN):**
   - A unique 10-character alphanumeric identifier formatted strictly as `AAAAA9999A`.
   - **Structure breakdown:**
     - **Characters 1 to 3 (AAA):** Alphabetic series from `AAA` to `ZZZ`.
     - **Character 4 (P):** Status / Entity category of the PAN holder:
       - `P` = Individual (Person)
       - `C` = Company
       - `H` = Hindu Undivided Family (HUF)
       - `F` = Partnership Firm / Limited Liability Partnership (LLP)
       - `A` = Association of Persons (AOP)
       - `T` = Trust
       - `B` = Body of Individuals (BOI)
       - `L` = Local Authority
       - `J` = Artificial Juridical Person
       - `G` = Government Agency
     - **Character 5 (A):** First character of the cardholder's surname (last name) for individuals, or legal entity name for non-individuals.
     - **Characters 6 to 9 (9999):** Sequential 4-digit number from `0001` to `9999`.
     - **Character 10 (A):** Alphabetic check digit.
2. **Cardholder Full Name:**
   - Printed in English capital letters below the national header.
3. **Father's Name / Parent's Name:**
   - Name of the cardholder's father (or mother if selected during application).
4. **Date of Birth (DOB) / Date of Incorporation:**
   - Formatted strictly as `DD/MM/YYYY`.
5. **Cardholder Signature:**
   - Physical or digitally uploaded specimen signature.
6. **Government Insignia & Hologram:**
   - "INCOME TAX DEPARTMENT / आयकर विभाग" and "GOVT. OF INDIA / भारत सरकार".
   - Official national hologram and security guilloche wavy line background patterns.

---

## 2. Card Templates & Variants

PAN cards exist in multiple officially valid physical and electronic versions:
1. **Older Physical PAN Cards (Pre-2017):**
   - Traditional plastic cards containing hologram, text, signature, and photo. Some business/entity PAN cards omit photographs entirely.
2. **Newer Enhanced Quick Response (QR) PAN Cards (Post-2017):**
   - Includes an enhanced QR code containing cardholder details, photograph, and digital signature for offline verification.
3. **e-PAN (Instant Digital PAN):**
   - Issued electronically via Aadhaar e-KYC. Features QR code, photo, and digital signature.
4. **Non-Photo PAN Templates:**
   - PAN cards issued to corporations, firms, trusts, or certain older individual issuances do NOT contain a photograph. Face verification MUST be marked as NOT APPLICABLE with zero risk penalty.

---

## 3. Official Verification Procedures

A thorough PAN verification process includes:
1. **PAN Format & Syntax Regex Validation:**
   - Match against standard regular expression `^[A-Z]{5}[0-9]{4}[A-Z]{1}$`.
2. **Entity & Surname Consistency Check:**
   - Cross-check the 4th character against the applicant type (e.g. `P` for individual applicants).
   - Cross-check the 5th character against the initial letter of the extracted surname.
3. **DOB & Date Format Consistency:**
   - Validate that the Date of Birth represents a plausible past date and is consistent with other applicant records.
4. **Income Tax Department / NSDL Database Verification:**
   - In production systems, query the NSDL / UTIITSL PAN verification API to confirm active status.

---

## 4. Common Inconsistencies & Diagnostic Guidelines

When analyzing PAN document captures:
1. **Missing Photo on Non-Photo Templates:**
   - If the card does not have a photo area or is a non-individual/older card, absence of a photo is completely expected and legitimate. It must not increase the risk score.
2. **Wear, Tear, and Reflection on Hologram:**
   - The metallic hologram frequently reflects camera flash, causing localized bright spots or high Error Level Analysis (ELA) scores. This is a common physical capture artifact, not digital tampering.
3. **Signs of Actual Forgery:**
   - Inconsistent fonts, mismatched letter spacing in the 10-digit PAN string, altered birth dates with visible background smudging, or 4th character mismatch with person status.
