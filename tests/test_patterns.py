import json

import pytest

from tenant_memory import AnonymizationError, InvalidInput, assert_anonymous, describe, skeleton_from_manifest

FORBIDDEN = ["brightside-dental", "harbor-realty"]


_VOCABULARY_KEYS = {"type", "image_side"}  # values drawn from the schema's own enums, not written by a client


def client_text(value):
    """Every string a client wrote: values only, skipping keys and schema enum values."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            if k not in _VOCABULARY_KEYS:
                yield from client_text(v)
    elif isinstance(value, list):
        for v in value:
            yield from client_text(v)


def test_skeleton_keeps_structure_and_drops_all_text(manifest):
    skeleton = skeleton_from_manifest(manifest)
    assert [s["type"] for s in skeleton] == [s["type"] for s in manifest["sections"]]
    assert skeleton[0] == {"type": "hero", "has_image": True, "has_subheading": True, "has_cta": True}
    assert skeleton[1] == {"type": "feature_grid", "items": 3, "columns": 3}

    # Nothing a client wrote may appear anywhere in the skeleton.
    written = [t for t in client_text(manifest["sections"]) if len(t) > 6]
    flat = json.dumps(skeleton).lower()
    assert written
    assert not [t for t in written if t.lower() in flat]


def test_description_is_generated_from_structure_only(manifest):
    text = describe("dental", "lead-generation", skeleton_from_manifest(manifest))
    assert text.startswith("industry dental; goal lead-generation; layout hero")
    assert "Book a consultation" not in text and "clinic" not in text


@pytest.mark.parametrize("bad", [{}, {"sections": []}, {"sections": "x"}, {"sections": [{"type": "carousel"}]}, {"sections": ["hero"]}, []])
def test_malformed_manifests_are_rejected(bad):
    with pytest.raises(InvalidInput):
        skeleton_from_manifest(bad)


@pytest.mark.parametrize(
    "value",
    ["owner@clinic.example", "https://brightside.example", "www.brightside.example", "call 0917 555 1234", "brightside dental", "Brightside-Dental"],
)
def test_identifying_strings_are_refused(value):
    with pytest.raises(AnonymizationError):
        assert_anonymous([value], FORBIDDEN)


def test_plain_structural_labels_pass():
    assert_anonymous(["dental", "lead-generation", "hero-then-grid"], FORBIDDEN)


def test_upsert_and_search_by_structure(patterns, manifest, clone):
    patterns.upsert("dental-home-1", "dental", "lead-generation", manifest, FORBIDDEN)
    short = clone(manifest)
    short["sections"] = [s for s in short["sections"] if s["type"] in ("hero", "cta_band")]
    patterns.upsert("dental-landing-2", "dental", "lead-generation", short, FORBIDDEN)
    patterns.upsert("realty-home-1", "realty", "listings", manifest, FORBIDDEN)

    hits = patterns.search("industry dental layout hero then cta band", industry="dental", limit=5)
    assert {h.pattern_id for h in hits} == {"dental-home-1", "dental-landing-2"}
    assert hits[0].pattern_id == "dental-landing-2"  # the shorter hero-and-cta layout is the closer match
    assert all(h.industry == "dental" for h in hits)


def test_the_stored_payload_contains_no_client_text(patterns, qdrant, manifest):
    patterns.upsert("dental-home-1", "dental", "lead-generation", manifest, FORBIDDEN)
    points, _ = qdrant.scroll("layout_patterns", with_payload=True, limit=10)
    assert len(points) == 1
    payload = points[0].payload
    assert set(payload) == {"pattern_id", "industry", "goal", "skeleton", "description"}
    stored = json.dumps(payload).lower()
    for text in client_text(manifest["sections"]):
        if len(text) > 6:
            assert text.lower() not in stored


def test_pattern_inputs_are_validated(patterns, manifest):
    with pytest.raises(InvalidInput):
        patterns.upsert("x", "dental", "lead-generation", manifest)  # id too short
    with pytest.raises(InvalidInput):
        patterns.upsert("ok-id", "Dental Clinic", "lead-generation", manifest)
    with pytest.raises(AnonymizationError):
        patterns.upsert("ok-id", "brightside-dental", "lead-generation", manifest, FORBIDDEN)


def test_delete_and_replace(patterns, manifest):
    patterns.upsert("dental-home-1", "dental", "lead-generation", manifest)
    patterns.upsert("dental-home-1", "dental", "lead-generation", manifest)
    assert len(patterns.search("dental", limit=10)) == 1
    patterns.delete("dental-home-1")
    assert patterns.search("dental", limit=10) == []
