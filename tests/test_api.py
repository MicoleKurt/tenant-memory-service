import pytest

from conftest import KEY_A, KEY_ADMIN, KEY_B, bearer

DENTAL = "Our clinic offers teeth whitening, dental implants and a free first consultation."
REALTY = "Harbor Realty lists waterfront condos and negotiates closing costs for sellers."


def put(api, key, doc, text, **extra):
    return api.put(f"/v1/documents/{doc}", json={"text": text, **extra}, headers=bearer(key))


# -- authentication ---------------------------------------------------------------------


def test_health_needs_no_key(api):
    assert api.get("/healthz").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer"}, {"Authorization": ""}])
def test_every_data_route_rejects_missing_or_unknown_keys(api, headers):
    for method, path, body in (
        ("put", "/v1/documents/x", {"text": "hi"}),
        ("delete", "/v1/documents/x", None),
        ("post", "/v1/search", {"query": "hi"}),
        ("get", "/v1/stats", None),
        ("post", "/v1/context", {"query": "hi"}),
        ("post", "/v1/patterns/search", {"query": "hi"}),
        ("post", "/v1/patterns", {"pattern_id": "ab", "industry": "ab", "goal": "ab", "manifest": {}}),
        ("delete", "/v1/admin/tenants/harbor-realty", None),
    ):
        r = api.request(method, path, json=body, headers=headers)
        assert r.status_code == 401, (method, path)
        assert r.headers["www-authenticate"] == "Bearer"


# -- tenant routes ----------------------------------------------------------------------


def test_tenants_cannot_see_each_others_documents(api):
    assert put(api, KEY_A, "services", DENTAL).json() == {"document_id": "services", "chunks": 1}
    assert put(api, KEY_B, "services", REALTY).status_code == 200

    hits_a = api.post("/v1/search", json={"query": "waterfront condos closing"}, headers=bearer(KEY_A)).json()["hits"]
    hits_b = api.post("/v1/search", json={"query": "teeth whitening"}, headers=bearer(KEY_B)).json()["hits"]
    assert [h["text"] for h in hits_a] == [DENTAL]
    assert [h["text"] for h in hits_b] == [REALTY]
    assert api.get("/v1/stats", headers=bearer(KEY_A)).json() == {"chunks": 1}


def test_there_is_no_way_to_name_a_tenant(api):
    """The tenant comes from the key. Extra fields or headers claiming another tenant change nothing."""
    put(api, KEY_B, "services", REALTY)
    r = api.post(
        "/v1/search?tenant_id=harbor-realty",
        json={"query": "waterfront condos", "tenant_id": "harbor-realty"},
        headers={**bearer(KEY_A), "X-Tenant-Id": "harbor-realty"},
    )
    assert r.status_code == 200 and r.json()["hits"] == []


def test_delete_document_is_scoped_to_the_key(api):
    put(api, KEY_A, "faq", DENTAL)
    put(api, KEY_B, "faq", REALTY)
    assert api.delete("/v1/documents/faq", headers=bearer(KEY_A)).json() == {"document_id": "faq", "chunks_removed": 1}
    assert api.get("/v1/stats", headers=bearer(KEY_B)).json() == {"chunks": 1}


def test_validation_errors_are_422(api):
    assert put(api, KEY_A, "bad id!", DENTAL).status_code in (404, 422)
    assert put(api, KEY_A, "ok", DENTAL, metadata={"Bad Key": 1}).status_code == 422
    assert api.post("/v1/search", json={"query": ""}, headers=bearer(KEY_A)).status_code == 422
    assert api.post("/v1/search", json={"query": "x", "limit": 500}, headers=bearer(KEY_A)).status_code == 422
    assert put(api, KEY_A, "ok", "   \n  ").status_code == 422


def test_oversized_documents_are_413(api):
    assert put(api, KEY_A, "big", "word " * 2000).status_code == 413


# -- admin and the shared tier ---------------------------------------------------------


def test_admin_key_cannot_read_or_write_private_memory(api):
    for method, path, body in (
        ("put", "/v1/documents/x", {"text": "hi"}),
        ("post", "/v1/search", {"query": "hi"}),
        ("get", "/v1/stats", None),
        ("post", "/v1/context", {"query": "hi"}),
    ):
        assert api.request(method, path, json=body, headers=bearer(KEY_ADMIN)).status_code == 403


def test_tenant_keys_cannot_curate_patterns_or_offboard(api, manifest):
    body = {"pattern_id": "dental-home-1", "industry": "dental", "goal": "lead-generation", "manifest": manifest}
    assert api.post("/v1/patterns", json=body, headers=bearer(KEY_A)).status_code == 403
    assert api.delete("/v1/patterns/dental-home-1", headers=bearer(KEY_A)).status_code == 403
    assert api.delete("/v1/admin/tenants/harbor-realty", headers=bearer(KEY_A)).status_code == 403


def test_admin_curates_patterns_and_tenants_search_them(api, manifest):
    body = {"pattern_id": "dental-home-1", "industry": "dental", "goal": "lead-generation", "manifest": manifest}
    created = api.post("/v1/patterns", json=body, headers=bearer(KEY_ADMIN))
    assert created.status_code == 201
    assert set(created.json()) == {"pattern_id", "industry", "goal", "skeleton", "description"}

    found = api.post("/v1/patterns/search", json={"query": "dental hero feature grid", "industry": "dental"}, headers=bearer(KEY_B)).json()
    assert [p["pattern_id"] for p in found["patterns"]] == ["dental-home-1"]


def test_patterns_that_name_a_client_are_refused(api, manifest):
    body = {"pattern_id": "ok-id", "industry": "harbor-realty", "goal": "listings", "manifest": manifest}
    r = api.post("/v1/patterns", json=body, headers=bearer(KEY_ADMIN))
    assert r.status_code == 422


def test_context_returns_both_tiers_separately_labelled(api, manifest):
    api.post("/v1/patterns", json={"pattern_id": "dental-home-1", "industry": "dental", "goal": "lead-generation", "manifest": manifest}, headers=bearer(KEY_ADMIN))
    put(api, KEY_A, "services", DENTAL)
    put(api, KEY_B, "services", REALTY)

    body = api.post("/v1/context", json={"query": "teeth whitening clinic", "industry": "dental"}, headers=bearer(KEY_A)).json()
    assert [h["tier"] for h in body["client"]] == ["client"]
    assert body["client"][0]["text"] == DENTAL
    assert [p["tier"] for p in body["patterns"]] == ["global"]
    assert "Harbor" not in str(body)


def test_offboarding_deletes_only_that_tenant(api):
    put(api, KEY_A, "services", DENTAL)
    put(api, KEY_B, "services", REALTY)
    r = api.delete("/v1/admin/tenants/brightside-dental", headers=bearer(KEY_ADMIN))
    assert r.json() == {"tenant_id": "brightside-dental", "chunks_removed": 1}
    assert api.get("/v1/stats", headers=bearer(KEY_A)).json() == {"chunks": 0}
    assert api.get("/v1/stats", headers=bearer(KEY_B)).json() == {"chunks": 1}
    assert api.delete("/v1/admin/tenants/Bad_Id", headers=bearer(KEY_ADMIN)).status_code == 422
