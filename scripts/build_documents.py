"""Write the company's non-markdown documents: a PDF contract, a Word memo, an Excel rate card
and an email thread.

They are generated so they stay consistent with data/company, and committed so the app never
needs this script. Run it again after changing the company data:

    uv run python scripts/build_documents.py
"""

import json
import mailbox
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path

from docx import Document
from fpdf import FPDF
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
COMPANY = ROOT / "data" / "company"
OUT = ROOT / "data" / "documents"

FICTIONAL = "Kestrel Works Pvt Ltd is a fictional company. This document is sample data."

CONTRACT_PAGES = [
    (
        "Supply Agreement CON-104: Kestrel Works and Lotus Peak Technologies",
        [
            FICTIONAL,
            (
                "This agreement is between Kestrel Works Pvt Ltd (the Buyer) and Lotus Peak "
                "Technologies (vendor V-104, the Supplier). It starts on 1 October 2026."
            ),
            (
                "The Supplier is on probation with the Buyer until 31 March 2027. During probation "
                "the Buyer reviews delivery and quality every quarter, and may end probation early "
                "by written notice."
            ),
            "The Supplier may deliver to the Buyer's Pune and Bengaluru offices only.",
        ],
    ),
    (
        "Pricing, delivery and warranty",
        [
            "Prices are fixed in Indian rupees until 31 March 2027 and exclude GST.",
            (
                "The Supplier will deliver within 6 calendar days of receiving a purchase order. For "
                "each day of late delivery the Buyer may deduct 0.5% of the order value, up to 5%."
            ),
            (
                "Laptops carry a 24-month on-site warranty. Faulty units must be replaced within "
                "3 working days."
            ),
        ],
    ),
    (
        "Order limits and termination",
        [
            (
                "While on probation, a single purchase order may not exceed INR 5,00,000 including "
                "GST. Orders may not be split to stay under this limit."
            ),
            (
                "Either party may end this agreement with 30 days' written notice. The Buyer may end "
                "it immediately if the Supplier is suspended under the Buyer's vendor policy."
            ),
            (
                "Signed for the Buyer by the Head of Procurement, and for the Supplier by its "
                "Director of Sales."
            ),
        ],
    ),
]

MEMO = (
    "MEM-011 Laptop refresh and returns, Q4 2026",
    "From the IT Operations Manager to all department heads, 2 November 2026.",
    [
        (
            "Standard laptops are replaced every 4 years and engineering laptops every 3 years. "
            "A laptop is replaced early only if IT confirms it cannot be repaired."
        ),
        (
            "When a replacement arrives, the old laptop must be returned to the Bengaluru IT store "
            "within 10 working days, so its data can be wiped and the device reused or recycled."
        ),
        (
            "Returned laptops in good condition are added to office stock and are offered before "
            "anything new is bought, in line with the procurement policy."
        ),
        "Questions about this memo go to the IT service desk.",
    ],
)

RATE_CARD_COLUMNS = [
    "Product code",
    "Product",
    "Vendor ID",
    "Vendor",
    "Unit price",
    "Currency",
    "Lead time (days)",
    "Warranty (months)",
]
WARRANTY_MONTHS = {"V-101": 36, "V-102": 24, "V-103": 24, "V-104": 24, "V-105": 12, "V-106": 12}

EMAILS = [
    (
        "Priya Nair, Engineering",
        "Finance Controller",
        "Network switches for Pune: only one supplier?",
        (
            "Hi, we need two 48-port switches for the Pune lab. The rate card lists only "
            "BlueMarine Imports for NET-SW48, and they bill in US dollars. Can we go ahead with a "
            "single supplier?"
        ),
    ),
    (
        "Finance Controller",
        "Priya Nair, Engineering",
        "Re: Network switches for Pune: only one supplier?",
        (
            "Yes. No approved Indian supplier carries this switch, so this is a sole-source "
            "purchase. Record that reason on the purchase order. Because it is billed in dollars, "
            "I will need to approve it myself, and the price must include the currency buffer."
        ),
    ),
    (
        "Procurement desk",
        "Priya Nair, Engineering",
        "Re: Network switches for Pune: only one supplier?",
        (
            "Noted. Please raise the request through the procurement assistant so the approvals "
            "are recorded."
        ),
    ),
]


def build_contract(path: Path) -> None:
    pdf = FPDF()
    pdf.set_title("Lotus Peak supply agreement")
    pdf.set_auto_page_break(auto=True, margin=20)
    for heading, paragraphs in CONTRACT_PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, heading)
        pdf.ln(4)
        pdf.set_font("Helvetica", size=11)
        for paragraph in paragraphs:
            pdf.multi_cell(0, 6, paragraph)
            pdf.ln(3)
    pdf.set_creation_date(datetime(2026, 10, 1, tzinfo=UTC))
    pdf.output(str(path))


def build_memo(path: Path) -> None:
    title, byline, paragraphs = MEMO
    document = Document()
    document.core_properties.title = title.partition(" ")[2]
    document.add_heading(title, level=1)
    document.sections[0].footer.paragraphs[0].text = FICTIONAL
    document.add_paragraph(byline)
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(str(path))


def build_rate_card(path: Path) -> None:
    products = {p["sku"]: p["name"] for p in json.loads((COMPANY / "products.json").read_text())}
    vendors = {v["id"]: v for v in json.loads((COMPANY / "vendors.json").read_text())}
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Q4 2026 rate card"
    sheet.append(RATE_CARD_COLUMNS)
    for offer in json.loads((COMPANY / "catalog.json").read_text()):
        vendor = vendors[offer["vendor_id"]]
        sheet.append(
            [
                offer["sku"],
                products[offer["sku"]],
                vendor["id"],
                vendor["name"],
                offer["unit_price"],
                vendor["currency"],
                offer["lead_time_days"],
                WARRANTY_MONTHS[vendor["id"]],
            ]
        )
    workbook.save(str(path))


def build_email_thread(path: Path) -> None:
    path.unlink(missing_ok=True)
    box = mailbox.mbox(str(path))
    for day, (sender, recipient, subject, body) in enumerate(EMAILS, start=3):
        message = EmailMessage()
        message["From"] = sender
        message["To"] = recipient
        message["Subject"] = subject
        message["Date"] = f"Tue, {day:02d} Nov 2026 10:00:00 +0530"
        message.set_content(body)
        box.add(message)
    box.flush()
    box.close()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    build_contract(OUT / "CON-104-lotus-peak-supply-agreement.pdf")
    build_memo(OUT / "MEM-011-laptop-refresh-and-returns.docx")
    build_rate_card(OUT / "PRL-Q4-vendor-rate-card.xlsx")
    build_email_thread(OUT / "EML-022-network-switch-sourcing.mbox")
    print(f"Wrote 4 documents to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
