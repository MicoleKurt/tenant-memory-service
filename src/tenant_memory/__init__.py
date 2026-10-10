"""Tenant-isolated vector memory for multi-client AI pipelines."""

from .chunking import chunk_text
from .embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from .errors import (
    AnonymizationError, AuthError, ConfigError, EmbeddingError, Forbidden, InvalidInput, IsolationError, TenantMemoryError,
)
from .keys import KeyRegistry, Principal, generate_key, hash_key
from .patterns import PatternHit, PatternStore, assert_anonymous, describe, skeleton_from_manifest
from .store import Hit, TenantScopedStore
from .tenant import TenantContext

__version__ = "0.1.0"
__author__ = "Micole Kurt T. Gonda"

__all__ = [
    "AnonymizationError", "AuthError", "ConfigError", "Embedder", "EmbeddingError", "Forbidden", "HashingEmbedder", "Hit",
    "InvalidInput", "IsolationError", "KeyRegistry", "OpenAIEmbedder", "PatternHit", "PatternStore", "Principal",
    "TenantContext", "TenantMemoryError", "TenantScopedStore", "assert_anonymous", "chunk_text", "describe",
    "generate_key", "hash_key", "skeleton_from_manifest",
]
