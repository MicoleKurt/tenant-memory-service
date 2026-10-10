"""HTTP API.

The tenant is never a parameter. It is derived from the bearer key on the server,
so there is no path, query or body field a client could change to reach another
tenant's memory.

    GET    /healthz                     liveness, no auth
    PUT    /v1/documents/{id}           store or replace a document        (tenant key)
    DELETE /v1/documents/{id}           remove a document                  (tenant key)
    POST   /v1/search                   search the caller's private memory (tenant key)
    GET    /v1/stats                    how much the caller has stored     (tenant key)
    POST   /v1/patterns/search          search shared layout patterns      (any key)
    POST   /v1/context                  both tiers in one call             (tenant key)
    POST   /v1/patterns                 curate a shared pattern            (admin key)
    DELETE /v1/patterns/{id}            remove a shared pattern            (admin key)
    DELETE /v1/admin/tenants/{tenant}   offboard a tenant                  (admin key)
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .errors import AuthError, EmbeddingError, InvalidInput, IsolationError
from .keys import KeyRegistry, Principal
from .patterns import PatternStore
from .store import TenantScopedStore
from .tenant import TenantContext

log = logging.getLogger("tenant_memory")


class DocumentIn(BaseModel):
    text: str = Field(min_length=1)
    metadata: dict[str, Any] | None = None


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    document_id: str | None = None


class PatternIn(BaseModel):
    pattern_id: str
    industry: str
    goal: str
    manifest: dict[str, Any]


class PatternSearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    industry: str | None = None
    limit: int = Field(default=3, ge=1, le=20)


class ContextIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    industry: str | None = None
    client_limit: int = Field(default=5, ge=1, le=50)
    pattern_limit: int = Field(default=3, ge=1, le=20)


def create_app(
    *,
    store: TenantScopedStore,
    patterns: PatternStore,
    registry: KeyRegistry,
    max_doc_chars: int = 200_000,
    extra_forbidden_terms: tuple[str, ...] = (),
) -> FastAPI:
    app = FastAPI(title="tenant-memory-service", version="0.1.0", docs_url=None, redoc_url=None)

    # -- error mapping ---------------------------------------------------------------

    @app.exception_handler(InvalidInput)
    async def _invalid(_: Request, exc: InvalidInput) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(EmbeddingError)
    async def _embedding(_: Request, exc: EmbeddingError) -> JSONResponse:
        log.warning("embedding failure: %s", exc)
        return JSONResponse({"detail": "the embedding provider failed"}, status_code=502)

    @app.exception_handler(IsolationError)
    async def _isolation(_: Request, exc: IsolationError) -> JSONResponse:
        log.error("ISOLATION VIOLATION BLOCKED: %s", exc)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    # -- authentication --------------------------------------------------------------

    def principal(authorization: str | None = Header(default=None)) -> Principal:
        try:
            return registry.authenticate(authorization)
        except AuthError:
            raise HTTPException(status_code=401, detail="invalid or missing API key", headers={"WWW-Authenticate": "Bearer"}) from None

    def tenant_only(p: Principal = Depends(principal)) -> TenantContext:
        if p.kind != "tenant" or p.tenant is None:
            raise HTTPException(status_code=403, detail="this route needs a tenant key")
        return p.tenant

    def admin_only(p: Principal = Depends(principal)) -> Principal:
        if p.kind != "admin":
            raise HTTPException(status_code=403, detail="this route needs the admin key")
        return p

    def forbidden_terms() -> list[str]:
        return [*registry.tenant_ids, *extra_forbidden_terms]

    # -- routes ----------------------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.put("/v1/documents/{document_id}")
    def put_document(document_id: str, body: DocumentIn, tenant: TenantContext = Depends(tenant_only)) -> dict[str, Any]:
        if len(body.text) > max_doc_chars:
            raise HTTPException(status_code=413, detail=f"document exceeds {max_doc_chars} characters")
        chunks = store.upsert_document(tenant, document_id, body.text, body.metadata)
        return {"document_id": document_id, "chunks": chunks}

    @app.delete("/v1/documents/{document_id}")
    def delete_document(document_id: str, tenant: TenantContext = Depends(tenant_only)) -> dict[str, Any]:
        return {"document_id": document_id, "chunks_removed": store.delete_document(tenant, document_id)}

    @app.post("/v1/search")
    def search(body: SearchIn, tenant: TenantContext = Depends(tenant_only)) -> dict[str, Any]:
        hits = store.search(tenant, body.query, body.limit, body.document_id)
        return {"hits": [asdict(h) for h in hits]}

    @app.get("/v1/stats")
    def stats(tenant: TenantContext = Depends(tenant_only)) -> dict[str, Any]:
        return {"chunks": store.count(tenant)}

    @app.post("/v1/patterns/search")
    def search_patterns(body: PatternSearchIn, _: Principal = Depends(principal)) -> dict[str, Any]:
        return {"patterns": [asdict(h) for h in patterns.search(body.query, body.industry, body.limit)]}

    @app.post("/v1/context")
    def context(body: ContextIn, tenant: TenantContext = Depends(tenant_only)) -> dict[str, Any]:
        client_hits = store.search(tenant, body.query, body.client_limit)
        pattern_hits = patterns.search(body.query, body.industry, body.pattern_limit)
        return {
            "client": [asdict(h) | {"tier": "client"} for h in client_hits],
            "patterns": [asdict(h) | {"tier": "global"} for h in pattern_hits],
        }

    @app.post("/v1/patterns", status_code=201)
    def put_pattern(body: PatternIn, _: Principal = Depends(admin_only)) -> dict[str, Any]:
        return patterns.upsert(body.pattern_id, body.industry, body.goal, body.manifest, forbidden_terms())

    @app.delete("/v1/patterns/{pattern_id}")
    def delete_pattern(pattern_id: str, _: Principal = Depends(admin_only)) -> dict[str, str]:
        patterns.delete(pattern_id)
        return {"deleted": pattern_id}

    @app.delete("/v1/admin/tenants/{tenant_id}")
    def offboard(tenant_id: str, _: Principal = Depends(admin_only)) -> dict[str, Any]:
        return {"tenant_id": tenant_id, "chunks_removed": store.delete_tenant(TenantContext.of(tenant_id))}

    return app
