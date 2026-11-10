"""Search every company document, structured or not, keeping the exact source of each passage.

Each document type is split into the smallest piece a person could check by eye: a policy
section, a PDF page, a Word paragraph, a spreadsheet row, an email message or a database record.
"""

import hashlib
import json
import mailbox
import re
from email.utils import parseaddr
from functools import cached_property
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from docx import Document
from fastembed import TextEmbedding
from openpyxl import load_workbook
from pydantic import BaseModel
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"
INDEX_DIR = ROOT / ".index"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "documents"

_SECTION_HEADING = re.compile(r"^## (\d+\.\d+) (.+)$", re.MULTILINE)
_DOC_FILE_NAME = re.compile(r"^([A-Z]{3}-[A-Z0-9]+)-(.+)$")

SOURCE_TYPES = {
    ".md": "Policy document",
    ".pdf": "PDF",
    ".docx": "Word document",
    ".xlsx": "Spreadsheet",
    ".mbox": "Email thread",
    ".json": "Database record",
}
STRUCTURED = {".xlsx", ".json"}


class Passage(BaseModel):
    """The smallest piece of a document that can be cited and checked."""

    doc_id: str
    doc_title: str
    unit: str = "section"
    section: str
    heading: str
    text: str
    source_file: str = ""

    @property
    def citation(self) -> str:
        return f"{self.doc_id} {self.unit} {self.section}"

    @property
    def source_type(self) -> str:
        return SOURCE_TYPES.get(Path(self.source_file).suffix, "Document")

    @property
    def structured(self) -> bool:
        return Path(self.source_file).suffix in STRUCTURED


class SearchHit(BaseModel):
    passage: Passage
    score: float


def _identify(path: Path) -> tuple[str, str]:
    """The ID and a fallback title, from the file name. Checked before the file is opened.

    'CON-104-lotus-peak-supply-agreement.pdf' -> ('CON-104', 'Lotus peak supply agreement').
    """
    match = _DOC_FILE_NAME.match(path.stem)
    if not match:
        raise ValueError(f"Document name must start with an ID such as CON-104-: {path.name}")
    return match.group(1), match.group(2).replace("-", " ").capitalize()


def _flat(text: str) -> str:
    return " ".join(text.split())


def split_policy(path: Path) -> list[Passage]:
    """Split a policy document into one passage per numbered section."""
    content = path.read_text(encoding="utf-8")
    title_line = content.splitlines()[0].removeprefix("# ").strip()
    doc_id, _, doc_title = title_line.partition(" ")
    matches = list(_SECTION_HEADING.finditer(content))
    passages = []
    for match, following in zip(matches, [*matches[1:], None]):
        end = following.start() if following else len(content)
        passages.append(
            Passage(
                doc_id=doc_id,
                doc_title=doc_title,
                section=match.group(1),
                heading=match.group(2),
                text=content[match.end() : end].strip(),
                source_file=path.name,
            )
        )
    return passages


def read_pdf(path: Path) -> list[Passage]:
    """One passage per page. The first line of each page is its heading."""
    doc_id, title = _identify(path)
    reader = PdfReader(path)
    title = (reader.metadata and reader.metadata.title) or title
    passages = []
    for number, page in enumerate(reader.pages, start=1):
        heading, _, body = (page.extract_text() or "").strip().partition("\n")
        if body.strip():
            passages.append(
                Passage(
                    doc_id=doc_id,
                    doc_title=title,
                    unit="page",
                    section=str(number),
                    heading=_flat(heading),
                    text=_flat(body),
                    source_file=path.name,
                )
            )
    return passages


def read_docx(path: Path) -> list[Passage]:
    """One passage per body paragraph, numbered as a reader would count them."""
    doc_id, title = _identify(path)
    document = Document(str(path))
    title = document.core_properties.title or title
    paragraphs = [
        p.text.strip()
        for p in document.paragraphs
        if p.text.strip() and not p.style.name.startswith(("Heading", "Title"))
    ]
    return [
        Passage(
            doc_id=doc_id,
            doc_title=title,
            unit="paragraph",
            section=str(number),
            heading=title,
            text=text,
            source_file=path.name,
        )
        for number, text in enumerate(paragraphs, start=1)
    ]


def read_xlsx(path: Path) -> list[Passage]:
    """One passage per row, numbered as the spreadsheet shows it, with the column names kept."""
    doc_id, title = _identify(path)
    sheet = load_workbook(path, read_only=True, data_only=True).active
    rows = sheet.iter_rows(values_only=True)
    columns = [str(c) for c in next(rows)]
    passages = []
    for number, row in enumerate(rows, start=2):
        if all(value is None for value in row):
            continue
        fields = "; ".join(f"{c}: {v}" for c, v in zip(columns, row) if v is not None)
        passages.append(
            Passage(
                doc_id=doc_id,
                doc_title=title,
                unit="row",
                section=str(number),
                heading=f"{title}, {row[0]}",
                text=fields,
                source_file=path.name,
            )
        )
    return passages


def read_mbox(path: Path) -> list[Passage]:
    """One passage per email, keeping who wrote it and when."""
    doc_id, title = _identify(path)
    passages = []
    for number, message in enumerate(mailbox.mbox(str(path)), start=1):
        body = message.get_payload(decode=True).decode(message.get_content_charset() or "utf-8")
        sender = parseaddr(message["From"])[0] or message["From"]
        recipient = parseaddr(message["To"])[0] or message["To"]
        passages.append(
            Passage(
                doc_id=doc_id,
                doc_title=title,
                unit="message",
                section=str(number),
                heading=message["Subject"],
                text=_flat(f"From {sender} to {recipient}, {message['Date']}. {body}"),
                source_file=path.name,
            )
        )
    return passages


def read_vendor_register(path: Path) -> list[Passage]:
    """One passage per vendor record from the company database."""
    return [
        Passage(
            doc_id="REG-VND",
            doc_title="Vendor Register",
            unit="record",
            section=vendor["id"],
            heading=vendor["name"],
            text=(
                f"Vendor {vendor['id']} {vendor['name']}. Status: {vendor['status']}. "
                f"Bills in {vendor['currency']}. Delivers to {', '.join(vendor['serves'])}."
            ),
            source_file=path.name,
        )
        for vendor in json.loads(path.read_text(encoding="utf-8"))
    ]


_READERS = {".pdf": read_pdf, ".docx": read_docx, ".xlsx": read_xlsx, ".mbox": read_mbox}


def load_passages(data_dir: Path = DATA_DIR) -> list[Passage]:
    """Every citable passage: policies, other documents, then database records."""
    passages = [
        p for path in sorted((data_dir / "policies").glob("*.md")) for p in split_policy(path)
    ]
    for path in sorted((data_dir / "documents").glob("*")):
        if path.suffix in _READERS:
            passages += _READERS[path.suffix](path)
    passages += read_vendor_register(data_dir / "company" / "vendors.json")
    return passages


def _fingerprint(passages: list[Passage]) -> str:
    joined = "".join(p.model_dump_json() for p in passages) + EMBEDDING_MODEL
    return hashlib.sha256(joined.encode()).hexdigest()


def _indexable_text(passage: Passage) -> str:
    return f"{passage.doc_title}. {passage.heading}. {passage.text}"


class KnowledgeBase:
    """A searchable index of every document, rebuilt only when a document changes."""

    def __init__(self, data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR):
        self._passages = load_passages(data_dir)
        self._client = chromadb.PersistentClient(
            path=str(index_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._sync()

    @cached_property
    def _embedder(self) -> TextEmbedding:
        return TextEmbedding(model_name=EMBEDDING_MODEL)

    def _sync(self):
        fingerprint = _fingerprint(self._passages)
        existing = self._client.get_or_create_collection(COLLECTION)
        if (existing.metadata or {}).get("fingerprint") == fingerprint:
            return existing

        self._client.delete_collection(COLLECTION)
        collection = self._client.create_collection(
            COLLECTION, metadata={"fingerprint": fingerprint, "hnsw:space": "cosine"}
        )
        texts = [_indexable_text(p) for p in self._passages]
        collection.add(
            ids=[p.citation for p in self._passages],
            documents=texts,
            embeddings=[e.tolist() for e in self._embedder.embed(texts)],
            metadatas=[p.model_dump() for p in self._passages],
        )
        return collection

    @property
    def size(self) -> int:
        """How many passages are indexed."""
        return self._collection.count()

    def search(self, query: str, k: int = 3, where: dict | None = None) -> list[SearchHit]:
        """Return the k most relevant passages, best first. Score is 1 for a perfect match."""
        if not query.strip():
            return []
        embedding = next(iter(self._embedder.query_embed(query))).tolist()
        result = self._collection.query(query_embeddings=[embedding], n_results=k, where=where)
        return [
            SearchHit(passage=Passage(**metadata), score=1 - distance)
            for metadata, distance in zip(result["metadatas"][0], result["distances"][0])
        ]

    def search_balanced(self, query: str, policies: int = 3, others: int = 2) -> list[SearchHit]:
        """The best policy sections plus the best passages from every other document.

        Searching both together let a memo that mentions laptops in every paragraph push the
        policy rule about laptops out of the results. Policy sections are the only "section" unit.
        """
        return self.search(query, policies, where={"unit": "section"}) + self.search(
            query, others, where={"unit": {"$ne": "section"}}
        )
