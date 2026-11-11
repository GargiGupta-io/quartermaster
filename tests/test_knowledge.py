"""Reading needs no model; the search tests use the real local embedding model."""

import pytest

from procurement_agent.knowledge import (
    DATA_DIR,
    POLICY_DIR,
    KnowledgeBase,
    load_passages,
    read_docx,
    read_mbox,
    read_pdf,
    read_xlsx,
    split_policy,
)

DOCUMENTS = DATA_DIR / "documents"


def test_split_keeps_source_and_section():
    passages = split_policy(POLICY_DIR / "POL-003-vendor-management.md")
    assert [p.section for p in passages] == ["3.1", "3.2", "3.3"]
    first = passages[0]
    assert first.doc_id == "POL-003"
    assert first.doc_title == "Vendor Management"
    assert first.heading == "Vendor status"
    assert first.citation == "POL-003 section 3.1"
    assert "Suspended" in first.text
    assert "Competitive quotes" not in first.text


def test_citations_are_unique_across_documents():
    citations = [p.citation for p in load_passages()]
    assert len(citations) == len(set(citations))


def test_pdf_is_split_by_page_with_the_page_heading():
    pages = read_pdf(DOCUMENTS / "CON-104-lotus-peak-supply-agreement.pdf")
    assert [p.citation for p in pages] == ["CON-104 page 1", "CON-104 page 2", "CON-104 page 3"]
    assert pages[0].doc_title == "Lotus Peak supply agreement"
    assert pages[1].heading == "Pricing, delivery and warranty"
    assert "24-month on-site warranty" in pages[1].text
    assert pages[1].source_type == "PDF"
    assert not pages[1].structured


def test_word_memo_counts_body_paragraphs_only():
    paragraphs = read_docx(DOCUMENTS / "MEM-011-laptop-refresh-and-returns.docx")
    assert paragraphs[0].citation == "MEM-011 paragraph 1"
    assert paragraphs[0].doc_title == "Laptop refresh and returns, Q4 2026"
    assert "replaced every 4 years" in paragraphs[1].text
    # The title is a heading and the disclaimer is a footer: neither is a paragraph to cite.
    assert not any("Laptop refresh" in p.text for p in paragraphs)
    assert not any("fictional" in p.text for p in paragraphs)


def test_spreadsheet_rows_keep_their_row_number_and_column_names():
    rows = read_xlsx(DOCUMENTS / "PRL-Q4-vendor-rate-card.xlsx")
    switch = next(r for r in rows if "NET-SW48" in r.text)
    assert switch.citation == "PRL-Q4 row 18"  # as Excel shows it, below the header row
    assert "Vendor ID: V-105" in switch.text
    assert "Currency: USD" in switch.text
    assert switch.source_type == "Spreadsheet"
    assert switch.structured


def test_email_thread_keeps_sender_and_order():
    messages = read_mbox(DOCUMENTS / "EML-022-network-switch-sourcing.mbox")
    assert [m.citation for m in messages] == [f"EML-022 message {n}" for n in (1, 2, 3)]
    assert messages[1].text.startswith("From Finance Controller to Priya Nair")
    assert "sole-source" in messages[1].text
    assert messages[1].source_type == "Email thread"


def test_vendor_records_come_from_the_database():
    records = {p.citation: p for p in load_passages() if p.doc_id == "REG-VND"}
    assert len(records) == 6
    quickline = records["REG-VND record V-106"]
    assert "Status: suspended" in quickline.text
    assert quickline.source_type == "Database record"
    assert quickline.structured


def test_every_document_type_is_loaded():
    types = {p.source_type for p in load_passages()}
    assert types == {
        "Policy document",
        "PDF",
        "Word document",
        "Spreadsheet",
        "Email thread",
        "Database record",
    }


def test_document_without_an_id_is_rejected(tmp_path):
    bad = tmp_path / "notes.pdf"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="must start with an ID"):
        read_pdf(bad)


@pytest.fixture(scope="module")
def knowledge(tmp_path_factory):
    return KnowledgeBase(index_dir=tmp_path_factory.mktemp("index"))


# Worded the way an employee would ask, avoiding the policy's own headings where possible.
QUESTIONS = [
    ("Who has to sign off on a ₹13 lakh order?", "POL-002 section 2.1"),
    ("Can I break one big order into several small ones?", "POL-002 section 2.3"),
    ("Does a dollar invoice need extra approval?", "POL-002 section 2.2"),
    ("Do we need to use laptops we already have before buying more?", "POL-001 section 1.2"),
    ("What if the order costs more than my department has left?", "POL-001 section 1.3"),
    ("Can we buy Amazon vouchers for the team?", "POL-001 section 1.5"),
    ("Is tax included when checking limits?", "POL-001 section 1.4"),
    ("Can we buy from a supplier that is suspended?", "POL-003 section 3.1"),
    ("How many quotations do I need to collect?", "POL-003 section 3.2"),
    ("Must we always go with the cheapest supplier?", "POL-003 section 3.3"),
    ("What is the maximum price for a standard laptop?", "POL-004 section 4.1"),
    ("A designer wants a more powerful laptop, what is the process?", "POL-004 section 4.2"),
    ("How many monitors can a new joiner get?", "POL-004 section 4.3"),
    ("An overseas seller is cheaper, can we use them?", "POL-005 section 5.1"),
    ("What exchange rate do I use for a USD price?", "POL-005 section 5.2"),
    ("How long do I have to claim back a taxi fare?", "POL-006 section 6.3"),
]


# Questions answered by documents other than the policies.
DOCUMENT_QUESTIONS = [
    ("How long is the warranty on Lotus Peak laptops?", "CON-104 page 2"),
    ("What penalty applies if Lotus Peak delivers late?", "CON-104 page 2"),
    ("What is the biggest single order we can give Lotus Peak?", "CON-104 page 3"),
    ("When does Lotus Peak's probation end?", "CON-104 page 1"),
    ("How often are engineering laptops replaced?", "MEM-011 paragraph 2"),
    ("Where do I send my old laptop after getting a new one?", "MEM-011 paragraph 3"),
    ("Did Finance agree the network switches can come from one supplier?", "EML-022 message 2"),
    ("Who sells the 48-port switch and at what price?", "PRL-Q4 row 18"),
    ("What warranty does Sahyadri give on monitors?", "PRL-Q4 row 11"),
    ("What does Konkan charge for an ergonomic chair?", "PRL-Q4 row 16"),
    ("Is Quickline Resellers allowed to supply us?", "REG-VND record V-106"),
    ("Which currency does BlueMarine bill in?", "REG-VND record V-105"),
]


@pytest.mark.parametrize(("question", "expected"), QUESTIONS + DOCUMENT_QUESTIONS)
def test_balanced_search_finds_the_right_passage(knowledge, question, expected):
    """What the agent gets: the 3 best policy sections plus the 3 best other passages."""
    hits = knowledge.search_balanced(question, policies=3, others=3)
    assert expected in [h.passage.citation for h in hits]


def test_balanced_search_always_includes_policy_sections(knowledge):
    hits = knowledge.search_balanced("laptop returns and refresh", policies=3, others=3)
    assert [h.passage.doc_id.startswith("POL-") for h in hits] == [True] * 3 + [False] * 3


def test_results_are_ordered_and_cite_a_source(knowledge):
    hits = knowledge.search("approval limits for purchases", k=3)
    assert len(hits) == 3
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert all(h.passage.citation for h in hits)


def test_blank_query_returns_nothing(knowledge):
    assert knowledge.search("   ") == []


def test_reopening_with_same_documents_reuses_the_index(tmp_path):
    first = KnowledgeBase(index_dir=tmp_path)
    second = KnowledgeBase(index_dir=tmp_path)
    assert "_embedder" in first.__dict__  # first open had to embed
    assert "_embedder" not in second.__dict__  # second open did not
