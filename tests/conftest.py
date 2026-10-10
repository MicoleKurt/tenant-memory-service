import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from tenant_memory import HashingEmbedder, KeyRegistry, PatternStore, TenantContext, TenantScopedStore, hash_key
from tenant_memory.api import create_app

FIXTURES = Path(__file__).resolve().parent / "fixtures"

KEY_A = "tms_key-for-brightside-dental-0000000000000000"
KEY_B = "tms_key-for-harbor-realty-000000000000000000000"
KEY_ADMIN = "tms_key-for-the-operator-0000000000000000000"

A = TenantContext.of("brightside-dental")
B = TenantContext.of("harbor-realty")
C = TenantContext.of("northwind-legal")


@pytest.fixture
def embedder():
    return HashingEmbedder(256)


@pytest.fixture
def qdrant():
    return QdrantClient(":memory:")


@pytest.fixture
def store(qdrant, embedder):
    s = TenantScopedStore(qdrant, embedder, "tenant_memory")
    s.ensure_collection()
    return s


@pytest.fixture
def patterns(qdrant, embedder):
    p = PatternStore(qdrant, embedder, "layout_patterns")
    p.ensure_collection()
    return p


@pytest.fixture
def manifest():
    return json.loads((FIXTURES / "homepage_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def clone():
    return copy.deepcopy


@pytest.fixture
def registry():
    return KeyRegistry(
        {hash_key(KEY_A): "brightside-dental", hash_key(KEY_B): "harbor-realty"},
        {hash_key(KEY_ADMIN)},
    )


@pytest.fixture
def api(store, patterns, registry):
    return TestClient(create_app(store=store, patterns=patterns, registry=registry, max_doc_chars=5000))


def bearer(key):
    return {"Authorization": f"Bearer {key}"}
