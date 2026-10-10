"""API keys: random bearer tokens, stored only as SHA-256 hashes.

A tenant key maps to exactly one tenant id on the server. The client never says
which tenant it is; the key says it. The admin key maps to no tenant, so it can
curate the shared pattern tier and offboard tenants but has no route that reads
a tenant's private memory.

Keys are 256-bit random values, so a plain SHA-256 is sufficient for storage
(there is nothing to brute-force); no password-hashing cost is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
from dataclasses import dataclass

from .errors import AuthError
from .tenant import TenantContext


def generate_key() -> str:
    return "tms_" + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Principal:
    kind: str  # "tenant" or "admin"
    tenant: TenantContext | None = None


class KeyRegistry:
    def __init__(self, tenant_key_hashes: dict[str, str], admin_key_hashes: frozenset[str] | set[str] = frozenset()) -> None:
        self._tenants = {h: TenantContext.of(t) for h, t in tenant_key_hashes.items()}
        self._admins = frozenset(admin_key_hashes)

    @property
    def tenant_ids(self) -> list[str]:
        return sorted({t.tenant_id for t in self._tenants.values()})

    def authenticate(self, authorization: str | None) -> Principal:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthError("missing bearer token")
        digest = hash_key(token.strip())
        if digest in self._admins:
            return Principal("admin")
        tenant = self._tenants.get(digest)
        if tenant is None:
            raise AuthError("unknown API key")
        return Principal("tenant", tenant)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tms-keygen", description="Generate an API key and the config entry that registers it.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("tenant_id", nargs="?", help="tenant id the key is bound to, e.g. brightside-dental")
    group.add_argument("--admin", action="store_true", help="generate an admin key instead")
    args = parser.parse_args()

    key = generate_key()
    digest = hash_key(key)
    if args.admin:
        print(f"admin key (shown once, give it to the operator): {key}")
        print(f'add to TMS_ADMIN_KEY_SHA256: {digest}')
    else:
        TenantContext.of(args.tenant_id)  # validate before printing anything
        print(f"API key for {args.tenant_id} (shown once, give it to the client): {key}")
        print(f'add to TMS_TENANT_KEYS: "{digest}": "{args.tenant_id}"')


if __name__ == "__main__":
    main()
