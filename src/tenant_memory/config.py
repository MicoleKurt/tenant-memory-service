"""Settings read from the environment. The service refuses to start on a bad config."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Mapping

from .errors import ConfigError, InvalidInput
from .tenant import TenantContext

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Settings:
    qdrant_url: str | None = None
    qdrant_path: str | None = None
    qdrant_api_key: str | None = None
    collection: str = "tenant_memory"
    pattern_collection: str = "layout_patterns"
    embedder: str = "hashing"
    embed_model: str = "text-embedding-3-large"
    embed_dimensions: int = 256
    openai_api_key: str | None = None
    tenant_keys: dict[str, str] = field(default_factory=dict)
    admin_keys: frozenset[str] = frozenset()
    forbidden_terms: tuple[str, ...] = ()
    max_doc_chars: int = 200_000

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        embedder = env.get("TMS_EMBEDDER", "hashing").lower()
        if embedder not in ("hashing", "openai"):
            raise ConfigError("TMS_EMBEDDER must be 'hashing' or 'openai'")
        default_dims = 1024 if embedder == "openai" else 256

        try:
            tenant_keys = json.loads(env.get("TMS_TENANT_KEYS", "{}") or "{}")
        except json.JSONDecodeError as exc:
            raise ConfigError("TMS_TENANT_KEYS must be a JSON object of {sha256: tenant_id}") from exc
        if not isinstance(tenant_keys, dict):
            raise ConfigError("TMS_TENANT_KEYS must be a JSON object of {sha256: tenant_id}")
        for digest, tenant in tenant_keys.items():
            if not _SHA256_RE.fullmatch(str(digest)):
                raise ConfigError("every key in TMS_TENANT_KEYS must be a lowercase SHA-256 hex digest")
            try:
                TenantContext.of(str(tenant))
            except InvalidInput as exc:
                raise ConfigError(f"TMS_TENANT_KEYS: {exc}") from None

        admin = frozenset(h.strip() for h in env.get("TMS_ADMIN_KEY_SHA256", "").split(",") if h.strip())
        if any(not _SHA256_RE.fullmatch(h) for h in admin):
            raise ConfigError("TMS_ADMIN_KEY_SHA256 must be comma-separated lowercase SHA-256 hex digests")
        if not tenant_keys and not admin:
            raise ConfigError("no API keys configured: set TMS_TENANT_KEYS and/or TMS_ADMIN_KEY_SHA256 (see `tms-keygen`)")

        openai_key = env.get("OPENAI_API_KEY") or None
        if embedder == "openai" and not openai_key:
            raise ConfigError("OPENAI_API_KEY is required when TMS_EMBEDDER=openai")

        return cls(
            qdrant_url=env.get("TMS_QDRANT_URL") or None,
            qdrant_path=env.get("TMS_QDRANT_PATH") or None,
            qdrant_api_key=env.get("TMS_QDRANT_API_KEY") or None,
            collection=env.get("TMS_COLLECTION", "tenant_memory"),
            pattern_collection=env.get("TMS_PATTERN_COLLECTION", "layout_patterns"),
            embedder=embedder,
            embed_model=env.get("TMS_EMBED_MODEL", "text-embedding-3-large"),
            embed_dimensions=int(env.get("TMS_EMBED_DIMENSIONS", default_dims)),
            openai_api_key=openai_key,
            tenant_keys={str(k): str(v) for k, v in tenant_keys.items()},
            admin_keys=admin,
            forbidden_terms=tuple(t.strip() for t in env.get("TMS_FORBIDDEN_TERMS", "").split(",") if t.strip()),
            max_doc_chars=int(env.get("TMS_MAX_DOC_CHARS", 200_000)),
        )
