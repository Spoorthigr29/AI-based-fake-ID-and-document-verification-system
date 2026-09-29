# Driving License Official Verification Knowledge Reference

**Document Type:** DRIVING_LICENSE
**Issuing Authority:** Ministry of Road Transport and Highways (MoRTH), Regional Transport Offices (RTO), State Governments of India
**Statutory Framework:** Motor Vehicles Act, 1988 & Central Motor Vehicles Rules, 1989

---

## 1. Expected Document Fields and Structure

An authentic Indian Driving License contains:
1. **Driving License Number:**
   - Standardized 16-character format: `SS-RRYYYYNNNNNNN` or `SSRR YYYYNNNNNNN`
   - Breakdown:
     - `SS`: 2-letter State/UT code (e.g. `DL`, `MH`, `KA`, `TN`, `UP`).
     - `RR`: 2-digit RTO office code.
     - `YYYY`: 4-digit Year of initial issuance.
     - `NNNNNNN`: 7-digit sequential unique number.
2. **Cardholder Demographics:**
   - Full Name, Son/Daughter/Spouse Name, Date of Birth (`DD-MM-YYYY`), Blood Group, Residential Address.
3. **Vehicle Class Authorizations:**
   - Authorized vehicle categories (e.g., `MCWG` - Motorcycle with Gear, `LMV` - Light Motor Vehicle, `TRANS` - Transport).
4. **Validity Periods:**
   - Non-Transport validity (up to age 50 or 20 years from issue), Transport validity (standard 5-year or 3-year Hazardous validity).
5. **Physical & Digital Indicators:**
   - Embedded Smart Chip or QR code, photograph, signature, state emblem.

---

## 2. Official Verification Procedures & Guidelines

1. **Format Validation:** Verify that the DL number adheres to the SARATHI nationwide format.
2. **Age & Class Consistency:** Validate that the cardholder was at least 18 years old on the issuance date for LMV/MCWG.
3. **SARATHI / Parivahan Database Lookup:** In production environments, query the national Parivahan portal.
