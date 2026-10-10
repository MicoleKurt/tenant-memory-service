import json

import pytest

from tenant_memory import AuthError, ConfigError, KeyRegistry, generate_key, hash_key
from tenant_memory.config import Settings
from tenant_memory.main import create_app

H1, H2 = hash_key("one"), hash_key("two")


def test_keys_are_random_and_long():
    a, b = generate_key(), generate_key()
    assert a != b and a.startswith("tms_") and len(a) >= 40


def test_registry_maps_a_key_to_exactly_one_tenant():
    reg = KeyRegistry({H1: "brightside-dental"}, {H2})
    assert reg.authenticate("Bearer one").tenant.tenant_id == "brightside-dental"
    assert reg.authenticate("bearer one").kind == "tenant"  # scheme is case-insensitive
    assert reg.authenticate("Bearer two").kind == "admin"


@pytest.mark.parametrize("header", [None, "", "Bearer", "Bearer ", "Basic one", "one", "Bearer wrong"])
def test_registry_rejects_everything_else(header):
    with pytest.raises(AuthError):
        KeyRegistry({H1: "brightside-dental"}).authenticate(header)


def env(**overrides):
    base = {"TMS_TENANT_KEYS": json.dumps({H1: "brightside-dental"})}
    base.update(overrides)
    return base


def test_settings_from_a_valid_environment():
    s = Settings.from_env(env(TMS_ADMIN_KEY_SHA256=H2, TMS_FORBIDDEN_TERMS="acme, globex"))
    assert s.tenant_keys == {H1: "brightside-dental"}
    assert s.admin_keys == {H2}
    assert s.forbidden_terms == ("acme", "globex")
    assert s.embedder == "hashing" and s.embed_dimensions == 256


@pytest.mark.parametrize(
    "bad",
    [
        {"TMS_TENANT_KEYS": "{}"},  # no keys at all
        {"TMS_TENANT_KEYS": "not json"},
        {"TMS_TENANT_KEYS": json.dumps({"short": "brightside-dental"})},
        {"TMS_TENANT_KEYS": json.dumps({H1: "Bad Tenant"})},
        {"TMS_ADMIN_KEY_SHA256": "nothex"},
        {"TMS_EMBEDDER": "mystery"},
        {"TMS_EMBEDDER": "openai"},  # needs OPENAI_API_KEY
    ],
)
def test_bad_settings_stop_the_service_from_starting(bad):
    environ = {"TMS_TENANT_KEYS": json.dumps({H1: "brightside-dental"})} if "TMS_TENANT_KEYS" not in bad else {}
    environ.update(bad)
    with pytest.raises(ConfigError):
        Settings.from_env(environ)


def test_the_app_factory_builds_a_working_service():
    from fastapi.testclient import TestClient

    key = generate_key()
    app = create_app(Settings.from_env({"TMS_TENANT_KEYS": json.dumps({hash_key(key): "brightside-dental"})}))
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    r = client.put("/v1/documents/hello", json={"text": "A first document."}, headers={"Authorization": f"Bearer {key}"})
    assert r.json() == {"document_id": "hello", "chunks": 1}
