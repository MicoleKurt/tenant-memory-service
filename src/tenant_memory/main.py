"""Application factory for uvicorn: ``uvicorn tenant_memory.main:create_app --factory``."""

from __future__ import annotations

from fastapi import FastAPI
from qdrant_client import QdrantClient

from .api import create_app as build_app
from .config import Settings
from .embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from .keys import KeyRegistry
from .patterns import PatternStore
from .store import TenantScopedStore


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "openai":
        return OpenAIEmbedder(settings.openai_api_key or "", model=settings.embed_model, dimensions=settings.embed_dimensions)
    return HashingEmbedder(settings.embed_dimensions)


def make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(":memory:")  # ephemeral: fine for trying it out, data is lost on restart


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    client = make_client(settings)
    embedder = make_embedder(settings)
    store = TenantScopedStore(client, embedder, settings.collection)
    patterns = PatternStore(client, embedder, settings.pattern_collection)
    store.ensure_collection()
    patterns.ensure_collection()
    registry = KeyRegistry(settings.tenant_keys, settings.admin_keys)
    return build_app(
        store=store,
        patterns=patterns,
        registry=registry,
        max_doc_chars=settings.max_doc_chars,
        extra_forbidden_terms=settings.forbidden_terms,
    )
