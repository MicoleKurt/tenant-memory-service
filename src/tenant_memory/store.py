"""Private, per-tenant memory in a single Qdrant collection.

Isolation rests on three independent mechanisms, because no single one should be
trusted alone:

1. **Server-side filter on every operation.** Methods take a ``TenantContext`` and
   build ``tenant_id == <tenant>`` themselves. There is no parameter through which a
   caller can pass its own filter, and the filter is applied to search, count,
   delete and the stale-chunk sweep alike.
2. **Tenant-scoped point ids.** Ids are ``uuid5(tenant, document, chunk)``, so two
   tenants uploading a document with the same name can never overwrite each other.
3. **A check on the way out.** Every point returned by a search is verified to carry
   the caller's tenant id; a mismatch raises ``IsolationError`` instead of returning
   data.

Qdrant's own multitenancy guidance is a single collection partitioned by a payload
key with a keyword index flagged ``is_tenant=True`` (co-locates a tenant's vectors
on disk and builds per-tenant HNSW links), rather than a collection per tenant.
That index is a performance feature, not a security boundary, which is why the
filter above exists.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from .chunking import chunk_text
from .embeddings import Embedder
from .errors import ConfigError, InvalidInput, IsolationError
from .tenant import TenantContext

_NAMESPACE = uuid.UUID("6f1a4f0e-2f64-5d8e-9a43-0c7b6a1e5d11")
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_META_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MAX_META_KEYS = 20
MAX_META_VALUE_CHARS = 300


@dataclass(frozen=True)
class Hit:
    document_id: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any]


def validate_document_id(document_id: str) -> str:
    if not isinstance(document_id, str) or not _DOC_ID_RE.fullmatch(document_id):
        raise InvalidInput("document id must be 1-128 characters of letters, digits, '.', '_' or '-'")
    return document_id


def validate_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Metadata is stored under its own ``meta`` key, so it can never shadow ``tenant_id``."""
    if not metadata:
        return {}
    if not isinstance(metadata, dict) or len(metadata) > MAX_META_KEYS:
        raise InvalidInput(f"metadata must be an object with at most {MAX_META_KEYS} keys")
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not _META_KEY_RE.fullmatch(key):
            raise InvalidInput(f"metadata key {key!r} must match [a-z][a-z0-9_]{{0,39}}")
        if isinstance(value, bool) or isinstance(value, (int, float)):
            clean[key] = value
        elif isinstance(value, str) and len(value) <= MAX_META_VALUE_CHARS:
            clean[key] = value
        else:
            raise InvalidInput(f"metadata value for {key!r} must be a number, a boolean or a string up to {MAX_META_VALUE_CHARS} characters")
    return clean


class TenantScopedStore:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str = "tenant_memory") -> None:
        self._client = client
        self._embedder = embedder
        self.collection = collection

    # -- setup -----------------------------------------------------------------------

    def ensure_collection(self) -> None:
        size = self._embedder.dimensions
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
            )
        else:
            existing = self._client.get_collection(self.collection).config.params.vectors
            existing_size = getattr(existing, "size", None)
            if existing_size is not None and existing_size != size:
                raise ConfigError(
                    f"collection '{self.collection}' holds {existing_size}-dimension vectors but the embedder produces {size}; "
                    "use a new collection name or re-embed"
                )
        # is_tenant: co-locate each tenant's vectors and build per-tenant HNSW links (Qdrant >= 1.11).
        self._client.create_payload_index(
            self.collection,
            field_name="tenant_id",
            field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
        )
        self._client.create_payload_index(self.collection, field_name="document_id", field_schema=models.PayloadSchemaType.KEYWORD)

    # -- filters ---------------------------------------------------------------------

    @staticmethod
    def _filter(tenant: TenantContext, *extra: models.Condition) -> models.Filter:
        if not isinstance(tenant, TenantContext):
            raise TypeError("a TenantContext is required")
        return models.Filter(
            must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant.tenant_id)), *extra]
        )

    @staticmethod
    def _doc(document_id: str) -> models.FieldCondition:
        return models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))

    @staticmethod
    def _point_id(tenant: TenantContext, document_id: str, index: int) -> str:
        return str(uuid.uuid5(_NAMESPACE, f"{tenant.tenant_id}\x1f{document_id}\x1f{index}"))

    # -- writes ----------------------------------------------------------------------

    def upsert_document(self, tenant: TenantContext, document_id: str, text: str, metadata: dict[str, Any] | None = None) -> int:
        """Store or replace a document. Returns the number of chunks now stored for it."""
        validate_document_id(document_id)
        meta = validate_metadata(metadata)
        chunks = chunk_text(text)
        if not chunks:
            raise InvalidInput("document text is empty")
        vectors = self._embedder.embed(chunks)
        points = [
            models.PointStruct(
                id=self._point_id(tenant, document_id, i),
                vector=vector,
                payload={"tenant_id": tenant.tenant_id, "document_id": document_id, "chunk_index": i, "text": chunk, "meta": meta},
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self._client.upsert(self.collection, points=points, wait=True)
        # A shorter re-upload must not leave the old tail behind.
        self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=self._filter(tenant, self._doc(document_id), models.FieldCondition(key="chunk_index", range=models.Range(gte=len(chunks))))
            ),
            wait=True,
        )
        return len(chunks)

    def delete_document(self, tenant: TenantContext, document_id: str) -> int:
        validate_document_id(document_id)
        flt = self._filter(tenant, self._doc(document_id))
        removed = self._client.count(self.collection, count_filter=flt, exact=True).count
        if removed:
            self._client.delete(self.collection, points_selector=models.FilterSelector(filter=flt), wait=True)
        return removed

    def delete_tenant(self, tenant: TenantContext) -> int:
        """Offboarding: remove everything a tenant has stored."""
        flt = self._filter(tenant)
        removed = self._client.count(self.collection, count_filter=flt, exact=True).count
        if removed:
            self._client.delete(self.collection, points_selector=models.FilterSelector(filter=flt), wait=True)
        return removed

    # -- reads -----------------------------------------------------------------------

    def count(self, tenant: TenantContext) -> int:
        return self._client.count(self.collection, count_filter=self._filter(tenant), exact=True).count

    def search(self, tenant: TenantContext, query: str, limit: int = 5, document_id: str | None = None) -> list[Hit]:
        if not query or not query.strip():
            raise InvalidInput("query is empty")
        if not 1 <= limit <= 50:
            raise InvalidInput("limit must be between 1 and 50")
        extra = [self._doc(validate_document_id(document_id))] if document_id else []
        vector = self._embedder.embed([query])[0]
        result = self._client.query_points(
            self.collection, query=vector, query_filter=self._filter(tenant, *extra), limit=limit, with_payload=True
        )
        hits: list[Hit] = []
        for point in result.points:
            payload = point.payload or {}
            if payload.get("tenant_id") != tenant.tenant_id:
                raise IsolationError("a result from another tenant reached the store boundary; request refused")
            hits.append(
                Hit(
                    document_id=payload["document_id"],
                    chunk_index=int(payload["chunk_index"]),
                    text=payload["text"],
                    score=float(point.score),
                    metadata=dict(payload.get("meta") or {}),
                )
            )
        return hits
