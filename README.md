# tenant-memory-service

**Tenant-isolated vector memory** for multi-client AI pipelines. One Qdrant collection holds every client's private knowledge (briefs, service pages, FAQs, brand voice), and a second, shared tier holds **anonymized layout patterns** that every client benefits from without any client's words crossing over.

> **Author:** Micole Kurt T. Gonda ([@MicoleKurt](https://github.com/MicoleKurt)) · MIT licensed
> Part of a three-repo toolkit for programmatic WordPress/Elementor production:
> [elementor-layout-compiler](https://github.com/MicoleKurt/elementor-layout-compiler) (manifest → `_elementor_data`) ·
> [wp-staging-bridge](https://github.com/MicoleKurt/wp-staging-bridge) (the authenticated plugin that writes the page) ·
> **tenant-memory-service** (this repo, the retrieval layer that feeds both).

## Two tiers

| Tier | Holds | Who can read it | How isolation works |
| --- | --- | --- | --- |
| **Client** (private) | Chunked, embedded client documents | Only the tenant whose key wrote them | Server-side tenant filter on every operation, tenant-scoped point ids, and a check on every result before it leaves the store |
| **Global** (shared) | Layout skeletons: section types, order, item counts, booleans | Any key | Built **by construction** from structure only. No heading, paragraph, quote, name, URL or alt text is ever read into it |

`POST /v1/context` returns both tiers in one call, labelled `client` and `global`, which is what a generation step needs: *what this client says* plus *what structurally works for this kind of page*.

## Isolation, concretely

Qdrant's [multitenancy guidance](https://qdrant.tech/documentation/guides/multiple-partitions/) is one collection partitioned by a payload key, with a keyword index flagged `is_tenant=True` (Qdrant 1.11+) so each tenant's vectors are co-located, rather than a collection per tenant. That index makes tenant-filtered search fast. **It does not enforce anything.** The enforcement is in this service:

1. **The tenant is never a request parameter.** It comes from the API key, resolved on the server. There is no path, query, header or body field a client can change to reach another tenant (tested: `?tenant_id=`, a `tenant_id` body field and an `X-Tenant-Id` header are all ignored).
2. **The store only accepts a `TenantContext`.** It builds `tenant_id == <tenant>` itself and applies it to search, count, delete and the stale-chunk sweep. It has no parameter for a caller-supplied filter, and it refuses a bare string. A test spies on every Qdrant call and fails if one is sent without the tenant pinned.
3. **Point ids are `uuid5(tenant, document, chunk)`.** Two clients uploading `faq` can never overwrite each other.
4. **Results are checked on the way out.** If a point from another tenant ever reaches the boundary (a test simulates a bug that drops the filter), the request fails with a 500 and returns nothing.
5. **Metadata cannot shadow the tenant.** Caller metadata lives under its own `meta` key and is type- and size-checked.
6. **Least-privilege keys.** Tenant keys read and write only their own memory. The admin key curates the shared tier and offboards tenants, and has **no route that reads private memory**. Keys are 256-bit random tokens stored only as SHA-256 hashes.

## Quick start

```bash
pip install -e .[dev]
pytest                                    # 103 tests, uses Qdrant's in-process mode

tms-keygen brightside-dental              # prints a key once + the hash line for TMS_TENANT_KEYS
tms-keygen --admin

export TMS_TENANT_KEYS='{"<sha256>": "brightside-dental"}'
export TMS_ADMIN_KEY_SHA256='<sha256>'
uvicorn tenant_memory.main:create_app --factory --port 8080
```

```bash
# a client stores and searches its own memory
curl -X PUT  localhost:8080/v1/documents/services -H "Authorization: Bearer $KEY" \
     -H "Content-Type: application/json" -d '{"text": "We offer teeth whitening and implants...", "metadata": {"source": "brief"}}'
curl -X POST localhost:8080/v1/context -H "Authorization: Bearer $KEY" \
     -H "Content-Type: application/json" -d '{"query": "hero copy for whitening offer", "industry": "dental"}'

# the operator curates a shared pattern from a layout manifest (only its structure is kept)
curl -X POST localhost:8080/v1/patterns -H "Authorization: Bearer $ADMIN" \
     -H "Content-Type: application/json" -d @pattern.json
```

With Docker and a real Qdrant: `docker compose up` (see `docker-compose.yml`).

## API

| Method | Route | Key | Purpose |
| --- | --- | --- | --- |
| `GET` | `/healthz` | none | Liveness |
| `PUT` | `/v1/documents/{id}` | tenant | Store or replace a document (chunked, embedded; a shorter re-upload removes the old tail) |
| `DELETE` | `/v1/documents/{id}` | tenant | Remove a document |
| `POST` | `/v1/search` | tenant | Search the caller's private memory, optionally within one document |
| `GET` | `/v1/stats` | tenant | Chunks stored |
| `POST` | `/v1/context` | tenant | Private hits plus shared patterns, labelled by tier |
| `POST` | `/v1/patterns/search` | any | Search shared layout patterns, optionally by industry |
| `POST` | `/v1/patterns` | admin | Add a pattern from a layout manifest (structure only) |
| `DELETE` | `/v1/patterns/{id}` | admin | Remove a pattern |
| `DELETE` | `/v1/admin/tenants/{tenant}` | admin | Offboard: delete everything a tenant stored |

Errors: `401` unknown key, `403` wrong kind of key, `413` document too large, `422` invalid input (with the reason), `502` embedding provider failure.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `TMS_TENANT_KEYS` | none | JSON `{"<sha256 of key>": "<tenant-id>"}` |
| `TMS_ADMIN_KEY_SHA256` | none | Comma-separated hashes. The service will not start with no keys at all |
| `TMS_QDRANT_URL` / `TMS_QDRANT_API_KEY` | none | Qdrant server. Without it: `TMS_QDRANT_PATH` (local, persisted) or in-memory (data lost on restart) |
| `TMS_EMBEDDER` | `hashing` | `hashing` (deterministic, offline) or `openai` (`text-embedding-3-large`, needs `OPENAI_API_KEY`) |
| `TMS_EMBED_DIMENSIONS` | 256 / 1024 | Must match the collection; a mismatch stops startup instead of corrupting it |
| `TMS_FORBIDDEN_TERMS` | none | Extra client names the shared tier must refuse (tenant ids are always refused) |
| `TMS_MAX_DOC_CHARS` | 200000 | Per-document limit |

## Status and limits (read this)

* The suite runs on Qdrant's **in-process** mode (`QdrantClient(":memory:")`), which honours filters exactly but ignores payload indexes. The `is_tenant` index and the Docker/Qdrant-server path are configured but **not exercised by CI**.
* The default `HashingEmbedder` captures word overlap, not meaning; it exists so the isolation logic is testable offline. Use `openai` (or add another `Embedder`) for real retrieval quality. The OpenAI client is tested against a mocked transport, not the live API.
* Not included, by design for this scope: rate limiting, key rotation endpoints, audit log storage, reranking. Each is a small addition behind the existing boundaries.

## License

MIT, © 2026 Micole Kurt T. Gonda
