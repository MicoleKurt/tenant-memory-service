"""Tier 2: an anonymized, shared memory of layout *structure*.

Tier 1 (``store.py``) is private to a client. This tier lets every client's
generation benefit from what works structurally (which sections, in which order,
how many items) without a single word of any client's content crossing over.

Anonymity is enforced by construction rather than by scrubbing:

* a pattern is built by ``skeleton_from_manifest``, which reads **only** section
  types, counts and booleans, never a heading, paragraph, quote, name, URL or alt text;
* the searchable description is generated from that skeleton by ``describe``, so
  there is no free-text field a client's copy could ride in on;
* the only caller-supplied strings are ``industry`` and ``goal``, which must be short
  slugs, and ``assert_anonymous`` rejects emails, links, phone-like numbers and any
  string containing a known client identifier.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from qdrant_client import QdrantClient, models

from .embeddings import Embedder
from .errors import AnonymizationError, ConfigError, InvalidInput

_NAMESPACE = uuid.UUID("0b8c2f3e-7a45-5c1d-8d2e-4f5a6b7c8d9e")
_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,40}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,80}$")
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+")
_LINK_RE = re.compile(r"(https?://|www\.|\.com\b|\.ph\b|\.org\b|\.net\b)", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\d[\s().+-]*){7,}")

SECTION_TYPES = ("hero", "feature_grid", "text_image", "cta_band", "testimonials", "faq")
_ITEM_SECTIONS = {"feature_grid", "testimonials", "faq"}
_MAX_SECTIONS = 30


def assert_anonymous(values: Iterable[str], forbidden_terms: Iterable[str] = ()) -> None:
    """Raise if any string could identify a person or a client."""
    terms = [t.lower().replace("-", " ") for t in forbidden_terms if len(t) >= 3]
    for value in values:
        if _EMAIL_RE.search(value) or _LINK_RE.search(value) or _PHONE_RE.search(value):
            raise AnonymizationError("patterns may not contain emails, links or phone-like numbers")
        flat = value.lower().replace("-", " ")
        for term in terms:
            if term in flat:
                raise AnonymizationError("patterns may not contain a known client identifier")


def skeleton_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce a layout manifest to its structure. No text from the manifest survives."""
    sections = manifest.get("sections") if isinstance(manifest, dict) else None
    if not isinstance(sections, list) or not 1 <= len(sections) <= _MAX_SECTIONS:
        raise InvalidInput(f"manifest.sections must be a list of 1-{_MAX_SECTIONS} sections")
    skeleton: list[dict[str, Any]] = []
    for section in sections:
        kind = section.get("type") if isinstance(section, dict) else None
        if kind not in SECTION_TYPES:
            raise InvalidInput(f"unsupported section type: {str(kind)[:40]!r}")
        entry: dict[str, Any] = {"type": kind}
        if kind in _ITEM_SECTIONS:
            items = section.get("items")
            entry["items"] = len(items) if isinstance(items, list) else 0
        if kind == "feature_grid" and section.get("columns") in (2, 3, 4):
            entry["columns"] = section["columns"]
        if kind in ("hero", "text_image"):
            entry["has_image"] = isinstance(section.get("image"), dict)
        if kind == "hero":
            entry["has_subheading"] = bool(section.get("subheading"))
        if kind == "text_image":
            entry["image_side"] = "right" if section.get("image_side") == "right" else "left"
        if kind in ("hero", "cta_band"):
            entry["has_cta"] = isinstance(section.get("cta"), dict)
        skeleton.append(entry)
    return skeleton


def describe(industry: str, goal: str, skeleton: list[dict[str, Any]]) -> str:
    """Deterministic, structure-only text that is embedded and searched."""
    parts = []
    for s in skeleton:
        detail = [f"{s['items']} items"] if "items" in s else []
        if "columns" in s:
            detail.append(f"{s['columns']} columns")
        if s.get("has_image"):
            detail.append("image")
        if s.get("has_subheading"):
            detail.append("subheading")
        if s.get("has_cta"):
            detail.append("call to action")
        if "image_side" in s:
            detail.append(f"image {s['image_side']}")
        parts.append(s["type"] + (f" ({', '.join(detail)})" if detail else ""))
    return f"industry {industry}; goal {goal}; layout {' then '.join(parts)}"


@dataclass(frozen=True)
class PatternHit:
    pattern_id: str
    industry: str
    goal: str
    skeleton: list[dict[str, Any]]
    description: str
    score: float


class PatternStore:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str = "layout_patterns") -> None:
        self._client = client
        self._embedder = embedder
        self.collection = collection

    def ensure_collection(self) -> None:
        size = self._embedder.dimensions
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(self.collection, vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE))
        else:
            existing = getattr(self._client.get_collection(self.collection).config.params.vectors, "size", None)
            if existing is not None and existing != size:
                raise ConfigError(f"collection '{self.collection}' holds {existing}-dimension vectors but the embedder produces {size}")
        self._client.create_payload_index(self.collection, field_name="industry", field_schema=models.PayloadSchemaType.KEYWORD)

    def upsert(self, pattern_id: str, industry: str, goal: str, manifest: dict[str, Any], forbidden_terms: Iterable[str] = ()) -> dict[str, Any]:
        if not _ID_RE.fullmatch(pattern_id or ""):
            raise InvalidInput("pattern id must be 2-81 characters of a-z, 0-9, '.', '_' or '-'")
        for label, value in (("industry", industry), ("goal", goal)):
            if not _SLUG_RE.fullmatch(value or ""):
                raise InvalidInput(f"{label} must be a short lowercase slug such as 'dental' or 'lead-generation'")
        assert_anonymous([pattern_id, industry, goal], forbidden_terms)
        skeleton = skeleton_from_manifest(manifest)
        description = describe(industry, goal, skeleton)
        payload = {
            "pattern_id": pattern_id,
            "industry": industry,
            "goal": goal,
            "skeleton": skeleton,
            "description": description,
        }
        vector = self._embedder.embed([description])[0]
        self._client.upsert(
            self.collection,
            points=[models.PointStruct(id=str(uuid.uuid5(_NAMESPACE, pattern_id)), vector=vector, payload=payload)],
            wait=True,
        )
        return payload

    def delete(self, pattern_id: str) -> None:
        self._client.delete(self.collection, points_selector=models.PointIdsList(points=[str(uuid.uuid5(_NAMESPACE, pattern_id))]), wait=True)

    def search(self, query: str, industry: str | None = None, limit: int = 3) -> list[PatternHit]:
        if not query or not query.strip():
            raise InvalidInput("query is empty")
        if not 1 <= limit <= 20:
            raise InvalidInput("limit must be between 1 and 20")
        flt = None
        if industry:
            if not _SLUG_RE.fullmatch(industry):
                raise InvalidInput("industry must be a short lowercase slug")
            flt = models.Filter(must=[models.FieldCondition(key="industry", match=models.MatchValue(value=industry))])
        vector = self._embedder.embed([query])[0]
        result = self._client.query_points(self.collection, query=vector, query_filter=flt, limit=limit, with_payload=True)
        return [
            PatternHit(
                pattern_id=p.payload["pattern_id"],
                industry=p.payload["industry"],
                goal=p.payload["goal"],
                skeleton=p.payload["skeleton"],
                description=p.payload["description"],
                score=float(p.score),
            )
            for p in result.points
        ]
