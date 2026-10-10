import random

import pytest

from tenant_memory import InvalidInput, IsolationError, TenantContext
from tenant_memory.store import TenantScopedStore, validate_metadata

from conftest import A, B, C

DENTAL = "Our clinic offers teeth whitening, dental implants and a free first consultation for new patients."
REALTY = "Harbor Realty lists waterfront condos, handles buyer viewings and negotiates closing costs for sellers."
LEGAL = "Northwind advises on contract disputes, employment claims and commercial lease negotiations."


def test_a_tenant_only_ever_finds_its_own_documents(store):
    store.upsert_document(A, "services", DENTAL)
    store.upsert_document(B, "services", REALTY)
    store.upsert_document(C, "services", LEGAL)

    # Query with the *other* tenants' exact wording: it must still come back empty-handed of their text.
    for tenant, own in ((A, DENTAL), (B, REALTY), (C, LEGAL)):
        for foreign in (DENTAL, REALTY, LEGAL):
            hits = store.search(tenant, foreign, limit=10)
            assert hits, "the tenant has its own document to return"
            assert {h.text for h in hits} == {own}


def test_randomised_queries_never_cross_tenants(store):
    words = "clinic dental condos legal lease implants closing contract waterfront patient seller claims".split()
    rng = random.Random(7)
    own_text = {A: DENTAL, B: REALTY, C: LEGAL}
    for tenant, text in own_text.items():
        for i in range(4):
            store.upsert_document(tenant, f"doc-{i}", text + f" Section {i}.")
    for _ in range(60):
        query = " ".join(rng.sample(words, 4))
        for tenant, text in own_text.items():
            hits = store.search(tenant, query, limit=50)
            assert len(hits) == 4  # exactly the caller's four documents, however the query is worded
            assert all(h.text.startswith(text) for h in hits)


def test_same_document_id_in_two_tenants_does_not_collide(store):
    store.upsert_document(A, "faq", DENTAL)
    store.upsert_document(B, "faq", REALTY)
    assert store.count(A) == 1 and store.count(B) == 1
    assert store.search(A, "waterfront condos", limit=5)[0].text == DENTAL
    assert store.search(B, "teeth whitening", limit=5)[0].text == REALTY


def test_deleting_a_document_only_touches_the_callers_copy(store):
    store.upsert_document(A, "faq", DENTAL)
    store.upsert_document(B, "faq", REALTY)
    assert store.delete_document(A, "faq") == 1
    assert store.count(A) == 0
    assert store.count(B) == 1
    assert store.delete_document(A, "faq") == 0  # nothing left, and nothing of B's was removed


def test_offboarding_removes_one_tenant_and_leaves_the_rest(store):
    store.upsert_document(A, "a1", DENTAL)
    store.upsert_document(A, "a2", DENTAL + " More.")
    store.upsert_document(B, "b1", REALTY)
    assert store.delete_tenant(A) == 2
    assert store.count(A) == 0
    assert store.count(B) == 1


def test_a_shorter_reupload_leaves_no_stale_tail(store):
    long_text = "\n\n".join(f"Paragraph {i}. " + "filler words " * 60 for i in range(8))
    first = store.upsert_document(A, "guide", long_text)
    assert first > 2
    second = store.upsert_document(A, "guide", "Only one short paragraph now.")
    assert second == 1
    assert store.count(A) == 1
    assert store.search(A, "filler words", limit=10)[0].text == "Only one short paragraph now."


def test_reupload_is_idempotent(store):
    n = store.upsert_document(A, "guide", DENTAL)
    store.upsert_document(A, "guide", DENTAL)
    assert store.count(A) == n


def test_metadata_cannot_override_the_tenant(store):
    store.upsert_document(A, "doc", DENTAL, metadata={"tenant_id": "harbor-realty", "source": "brief"})
    assert store.count(B) == 0
    hit = store.search(A, "teeth whitening")[0]
    assert hit.metadata == {"tenant_id": "harbor-realty", "source": "brief"}  # kept as plain data under "meta"
    assert store.search(B, "teeth whitening") == []


@pytest.mark.parametrize(
    "metadata",
    [{"Bad Key": 1}, {"k": {"nested": 1}}, {"k": ["a"]}, {"k": "x" * 301}, {f"k{i}": i for i in range(21)}, {"": 1}],
)
def test_bad_metadata_is_rejected(metadata):
    with pytest.raises(InvalidInput):
        validate_metadata(metadata)


@pytest.mark.parametrize("bad", ["", "../etc", "a b", "x" * 129, "-leading", "semi;colon"])
def test_bad_document_ids_are_rejected(store, bad):
    with pytest.raises(InvalidInput):
        store.upsert_document(A, bad, DENTAL)


@pytest.mark.parametrize("bad", ["", "A", "-x", "x" * 64, "a_b", "a b", "ab/cd"])
def test_tenant_ids_are_validated(bad):
    with pytest.raises(InvalidInput):
        TenantContext.of(bad)


def test_the_store_refuses_a_bare_string_for_a_tenant(store):
    with pytest.raises(TypeError):
        store.count("brightside-dental")  # type: ignore[arg-type]


def test_search_validates_its_inputs(store):
    with pytest.raises(InvalidInput):
        store.search(A, "   ")
    with pytest.raises(InvalidInput):
        store.search(A, "ok", limit=0)
    with pytest.raises(InvalidInput):
        store.search(A, "ok", limit=51)


def test_document_scoped_search(store):
    store.upsert_document(A, "services", DENTAL)
    store.upsert_document(A, "about", "We were founded in 1998 by two sisters.")
    hits = store.search(A, "clinic", limit=10, document_id="about")
    assert {h.document_id for h in hits} == {"about"}


def test_a_foreign_point_at_the_boundary_fails_closed(store, qdrant, monkeypatch):
    """Belt and braces: even if the filter were somehow bypassed, nothing is returned."""
    store.upsert_document(B, "faq", REALTY)
    real = qdrant.query_points

    def leaky(collection, **kwargs):
        kwargs["query_filter"] = None  # simulate a bug that drops the filter
        return real(collection, **kwargs)

    monkeypatch.setattr(qdrant, "query_points", leaky)
    with pytest.raises(IsolationError):
        store.search(A, "waterfront condos")


def test_every_operation_carries_the_tenant_filter(store, qdrant, monkeypatch):
    """Record the filters sent to Qdrant and check each one pins tenant_id."""
    seen = []

    def spy(name):
        real = getattr(qdrant, name)

        def wrapper(*args, **kwargs):
            seen.append((name, kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(qdrant, name, wrapper)

    for name in ("query_points", "count", "delete"):
        spy(name)

    store.upsert_document(A, "doc", DENTAL)
    store.search(A, "clinic")
    store.count(A)
    store.delete_document(A, "doc")
    store.delete_tenant(A)

    def tenant_pinned(flt):
        return flt is not None and any(getattr(c, "key", None) == "tenant_id" and c.match.value == "brightside-dental" for c in flt.must)

    assert seen, "expected calls to be recorded"
    for name, kwargs in seen:
        flt = kwargs.get("query_filter") or kwargs.get("count_filter")
        if flt is None and "points_selector" in kwargs:
            flt = kwargs["points_selector"].filter
        assert tenant_pinned(flt), f"{name} was sent without a tenant filter"


def test_collection_setup_is_idempotent_and_checks_dimensions(qdrant, embedder):
    from tenant_memory import ConfigError, HashingEmbedder

    s = TenantScopedStore(qdrant, embedder, "tm")
    s.ensure_collection()
    s.ensure_collection()
    with pytest.raises(ConfigError):
        TenantScopedStore(qdrant, HashingEmbedder(64), "tm").ensure_collection()
