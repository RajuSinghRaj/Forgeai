"""Generate clean, multi-page synthetic PDF documents for ForgeAI Phase 6 demos."""

from __future__ import annotations

from pathlib import Path


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_multipage_pdf(path: Path, pages: list[list[str]]) -> None:
    """Write a valid multi-page PDF with Helvetica text lines readable by pypdf."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Object 1: Catalog, Object 2: Pages, Object 3: Font Helvetica
    # Each page gets 2 objects: Page object and Contents stream object.
    objects: list[bytes] = []
    page_obj_ids = [4 + idx * 2 for idx in range(len(pages))]
    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)

    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(f"2 0 obj << /Type /Pages /Kids [{kids_str}] /Count {len(pages)} >> endobj\n".encode("latin-1"))
    objects.append(b"3 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")

    for idx, lines in enumerate(pages):
        page_id = 4 + idx * 2
        content_id = page_id + 1
        stream_cmds = ["BT", "/F1 10 Tf", "50 750 Td", "14 TL"]
        for line in lines:
            safe = _pdf_escape(line.encode("latin-1", errors="replace").decode("latin-1"))
            stream_cmds.append(f"({safe}) Tj T*")
        stream_cmds.append("ET")
        stream_bytes = "\n".join(stream_cmds).encode("latin-1")

        objects.append(
            (
                f"{page_id} 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >> endobj\n"
            ).encode("latin-1")
        )
        objects.append(
            f"{content_id} 0 obj << /Length {len(stream_bytes)} >> stream\n".encode("latin-1")
            + stream_bytes
            + b"\nendstream endobj\n"
        )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    path.write_bytes(bytes(pdf))


def build_all_pdfs(root: Path) -> list[Path]:
    hr_dir = root / "demos" / "hr_policy" / "documents"
    ship_dir = root / "demos" / "ship_spares" / "documents"

    hr_leave = hr_dir / "01_leave_and_attendance_policy.pdf"
    write_multipage_pdf(
        hr_leave,
        [
            [
                "NORTHSTAR MARITIME SYSTEMS (FICTIONAL) - SYNTHETIC DEMO DATA",
                "01. Leave and Attendance Policy | Effective Date: 2026-01-01",
                "",
                "Notice: This is synthetic demonstration data, not a real company policy.",
                "",
                "Section 1: Annual Leave Entitlement",
                "- Full-time employees accrue 20 paid annual leave days per calendar year",
                "  (1.67 days accrued per completed month of service).",
                "- Up to 5 unused annual leave days may carry over into Q1 (until March 31)",
                "  of the following calendar year; days beyond 5 expire unless local law applies.",
                "- Leave requests of 1 to 3 consecutive days require 5 business days notice.",
                "- Leave requests of 4 or more days require 15 business days advance notice.",
                "",
                "Section 2: Sick Leave and Medical Certification",
                "- Employees receive 10 paid sick leave days per calendar year.",
                "- Medical certification from a licensed practitioner is required for any sick",
                "  leave absence exceeding 3 consecutive working days.",
            ],
            [
                "01. Leave and Attendance Policy (Page 2) - SYNTHETIC DEMO DATA",
                "",
                "Section 3: Parental Leave",
                "- Primary caregivers are eligible for 16 weeks of paid parental leave after",
                "  completing 6 months of continuous employment.",
                "- Secondary caregivers receive 4 weeks of paid parental leave under the same",
                "  6-month tenure eligibility rule.",
                "",
                "Section 4: Core Working Hours and Office Attendance",
                "- Standard working hours are 40 hours per week, Monday through Friday.",
                "- Mandatory core collaboration hours are 10:00 to 16:00 local office time.",
                "- Employees working on-site must badge in at the office desk by 10:15.",
                "- Exception: For approved remote days, Section 4 of the Remote and Hybrid",
                "  Work Policy overrides the physical office badge check-in requirement.",
            ],
        ],
    )

    hr_remote = hr_dir / "02_remote_and_hybrid_work_policy.pdf"
    write_multipage_pdf(
        hr_remote,
        [
            [
                "NORTHSTAR MARITIME SYSTEMS (FICTIONAL) - SYNTHETIC DEMO DATA",
                "02. Remote and Hybrid Work Policy | Effective Date: 2026-01-01",
                "",
                "Notice: This is synthetic demonstration data, not a real company policy.",
                "",
                "Section 1: Standard Hybrid Schedule",
                "- Eligible full-time employees may work remotely up to 2 days per week",
                "  (typically Tuesday and Thursday, or as agreed with their manager).",
                "- At least 3 days per week must be worked from the assigned office hub.",
                "",
                "Section 2: Probation Eligibility Restriction",
                "- New joiners are NOT eligible for routine weekly hybrid remote work during",
                "  their initial 90-calendar-day probation period (see Onboarding Policy).",
                "- Manager-approved ad-hoc remote days during probation are limited to urgent",
                "  IT or safety incidents documented in HR tickets.",
            ],
            [
                "02. Remote and Hybrid Work Policy (Page 2) - SYNTHETIC DEMO DATA",
                "",
                "Section 3: Temporary Full-Remote Exception",
                "- Confirmed regular employees may request up to 10 business days per calendar",
                "  year of temporary full-remote work for documented family-care needs.",
                "- This exception requires written approval from both the direct manager and VP.",
                "",
                "Section 4: Core Hours and Attendance Precedence",
                "- On approved remote days, employees must remain reachable online during core",
                "  hours (10:00 to 16:00 local time) instead of physical desk badge check-in.",
                "- Precedence Rule: This clause explicitly supersedes Section 4 physical badge",
                "  check-in rules in 01_leave_and_attendance_policy.pdf on approved remote days.",
                "- Home-office equipment support is governed by the USD 600 one-time allowance",
                "  in 03_benefits_and_reimbursement_policy.md after 30 days of active service.",
            ],
        ],
    )

    ship_a = ship_dir / "supplier_a_nordic_marine_specs.pdf"
    write_multipage_pdf(
        ship_a,
        [
            [
                "NORDIC MARINE SPARES GMBH (SUP-NORDIC) - SYNTHETIC DEMO DATA",
                "Technical & Commercial Specification Sheet | Revision 2026.1",
                "",
                "Notice: Synthetic demonstration document. Not a real supplier quotation.",
                "",
                "1. Supplier Profile & Commercial Terms",
                "- Supplier ID: SUP-NORDIC | Dispatch Hub: Hamburg / Rotterdam Port Depot",
                "- Standard Incoterms: CIF Rotterdam or EXW Hamburg Warehouse.",
                "- Indicative Lead Times: 14 calendar days via standard sea/road freight;",
                "  5 calendar days via expedited air courier (adds 12% air surcharge on unit total).",
                "- Important: Quoted catalogue prices and lead times are indicative only.",
                "  No order is confirmed until a formal Purchase Order (PO) is acknowledged.",
                "",
                "2. Mechanical Seals (MV-101 & MV-102)",
                "- MV-101 (Part NMS-65-ME): 65mm silicon-carbide face mechanical seal for main",
                "  engine cooling water pumps. Max pressure 16 bar, max temp 135C. MOQ: 4 units.",
                "- MV-102 (Part NMS-40-AB): 40mm carbon-ceramic seal for auxiliary bilge pumps.",
                "  Max pressure 8 bar, max temp 95C. MOQ: 2 units. NOT interchangeable with MV-101.",
            ],
            [
                "SUP-NORDIC Technical & Commercial Sheet (Page 2) - SYNTHETIC DEMO DATA",
                "",
                "3. Cooling Valves, Gaskets, Bearings & Electrical (SUP-NORDIC)",
                "- MV-105 (NMS-TW-82): Wax-element 82C thermostat; MOQ 2 units; 10-day lead time.",
                "- MV-108 (NMS-FWG-10): EPDM plate gasket 10-pack for freshwater generators;",
                "  MOQ 2 packs; standard 14-day sea lead time.",
                "- BRG-501 (NMS-TCB-501): Turbocharger thrust bearing kit; class-certified;",
                "  MOQ 1 kit; 14-day standard lead time; 5-day expedited air available.",
                "- BRG-502 (NMS-ST-320): Stern tube 320mm Viton lip ring; MOQ 2 rings; 21-day",
                "  special-molding lead time (no air-expedite shortcut available).",
                "- PMP-601 (NMS-IMP-601): Bronze ballast impeller; MOQ 1 unit; 14-day lead time.",
                "- ELC-702 (NMS-AVR-400): AVR-M400 voltage regulator; MOQ 1 unit; 14-day lead time.",
            ],
        ],
    )

    ship_b = ship_dir / "supplier_b_pacific_propulsion_specs.pdf"
    write_multipage_pdf(
        ship_b,
        [
            [
                "PACIFIC PROPULSION PARTS LTD (SUP-PACIFIC) - SYNTHETIC DEMO DATA",
                "Technical & Commercial Specification Sheet | Revision 2026.1",
                "",
                "Notice: Synthetic demonstration document. Not a real supplier quotation.",
                "",
                "1. Supplier Profile & Lead Times",
                "- Supplier ID: SUP-PACIFIC | Dispatch Hub: Jurong Marine Logistics Park, Singapore",
                "- Standard Lead Time: 10 calendar days FOB Singapore for stocked supplier SKUs.",
                "- Air Courier Option: 4 business days to major Asian/European ports (+15% freight).",
                "- Distinction Rule: Warehouse on-hand stock (`stock_qty` in inventory_catalogue.csv)",
                "  is immediately issuable; supplier replenishment requires PO + MOQ + lead time.",
                "",
                "2. Filtration Elements & Compatibility Warnings",
                "- FL-201 (Part PPP-FO-10M): 10-micron pleated fuel oil fine filter. MOQ: 6 units.",
                "  Rated for MGO/MDO fuel systems up to 12 bar differential.",
                "- FL-202 (Part PPP-LO-25M): 25-micron lubricating oil filter. MOQ: 6 units.",
                "  WARNING: FL-201 (10-micron fuel) and FL-202 (25-micron lube) have different",
                "  bypass spring ratings and micron ratings; they are NOT interchangeable.",
                "- FL-205 (Part PPP-HY-05): Hydraulic steering return cartridge; MOQ: 4 units; 10-day lead.",
            ],
            [
                "SUP-PACIFIC Technical & Commercial Sheet (Page 2) - SYNTHETIC DEMO DATA",
                "",
                "3. Fuel Injection, Sensors, Relays & Custom Assemblies",
                "- INJ-301 (PPP-INJ-D): Type-D nozzle tip, 8x0.32mm holes; MOQ 4 units; 10-day lead.",
                "- INJ-302 (PPP-INJ-AX): Auxiliary generator injector assembly; MOQ 2 units; 12-day lead.",
                "- SNS-401 (PPP-EGT-K45): Type-K 450mm exhaust thermocouple, 0-900C; MOQ 2 units.",
                "- SNS-402 (PPP-SPT-06): 0-6 Bar scavenge air transmitter, 4-20mA output; MOQ 2 units.",
                "- PMP-602 (PPP-FTP-RV): Gear pump relief valve set at 6.5 bar; MOQ 2 units.",
                "- ELC-701 (PPP-RLY-24): 24V DC alarm relay module; MOQ 5 units.",
                "- VA-900 (PPP-VA-900): Custom High-Pressure Overboard Discharge Valve Assembly.",
                "  Status: Built to vessel hull drawing only. No standard catalogue price exists;",
                "  unit price and lead time require a formal Request for Quotation (RFQ).",
            ],
        ],
    )

    return [hr_leave, hr_remote, ship_a, ship_b]


if __name__ == "__main__":
    created = build_all_pdfs(Path(__file__).resolve().parent.parent)
    for item in created:
        print("Created PDF:", item)
