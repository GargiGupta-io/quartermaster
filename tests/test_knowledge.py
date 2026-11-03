"""Splitting needs no model; the search tests use the real local embedding model."""

import pytest

from procurement_agent.knowledge import POLICY_DIR, KnowledgeBase, load_passages, split_policy


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


@pytest.mark.parametrize(("question", "expected"), QUESTIONS)
def test_search_finds_the_right_section_in_top_three(knowledge, question, expected):
    hits = knowledge.search(question, k=3)
    assert expected in [h.passage.citation for h in hits]


def test_results_are_ordered_and_cite_a_source(knowledge):
    hits = knowledge.search("approval limits for purchases", k=3)
    assert len(hits) == 3
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert all(h.passage.doc_id.startswith("POL-") for h in hits)


def test_blank_query_returns_nothing(knowledge):
    assert knowledge.search("   ") == []


def test_reopening_with_same_documents_reuses_the_index(tmp_path):
    first = KnowledgeBase(index_dir=tmp_path)
    second = KnowledgeBase(index_dir=tmp_path)
    assert "_embedder" in first.__dict__  # first open had to embed
    assert "_embedder" not in second.__dict__  # second open did not
