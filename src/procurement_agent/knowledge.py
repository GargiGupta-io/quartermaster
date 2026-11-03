"""Search the company policy documents, keeping the source of every passage."""

import hashlib
import re
from functools import cached_property
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastembed import TextEmbedding
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "data" / "policies"
INDEX_DIR = ROOT / ".index"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "policies"

_SECTION_HEADING = re.compile(r"^## (\d+\.\d+) (.+)$", re.MULTILINE)


class Passage(BaseModel):
    """One numbered policy section, the smallest unit that can be cited."""

    doc_id: str
    doc_title: str
    section: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id} section {self.section}"


class SearchHit(BaseModel):
    passage: Passage
    score: float


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
            )
        )
    return passages


def load_passages(policy_dir: Path = POLICY_DIR) -> list[Passage]:
    return [p for path in sorted(policy_dir.glob("*.md")) for p in split_policy(path)]


def _fingerprint(passages: list[Passage]) -> str:
    joined = "".join(p.model_dump_json() for p in passages) + EMBEDDING_MODEL
    return hashlib.sha256(joined.encode()).hexdigest()


def _indexable_text(passage: Passage) -> str:
    return f"{passage.doc_title}. {passage.heading}. {passage.text}"


class KnowledgeBase:
    """A searchable index of policy passages, rebuilt only when the documents change."""

    def __init__(self, policy_dir: Path = POLICY_DIR, index_dir: Path = INDEX_DIR):
        self._passages = load_passages(policy_dir)
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

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        """Return the k most relevant passages, best first. Score is 1 for a perfect match."""
        if not query.strip():
            return []
        embedding = next(iter(self._embedder.query_embed(query))).tolist()
        result = self._collection.query(query_embeddings=[embedding], n_results=k)
        return [
            SearchHit(passage=Passage(**metadata), score=1 - distance)
            for metadata, distance in zip(result["metadatas"][0], result["distances"][0])
        ]
