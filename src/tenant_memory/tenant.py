"""The tenant context that every private-memory operation must carry.

The store's methods take a ``TenantContext``, never a bare string and never a
caller-built filter. A tenant id can only be produced by ``TenantContext.of``,
which validates it, and the API only ever builds one from the authenticated key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import InvalidInput

_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str

    @classmethod
    def of(cls, tenant_id: str) -> "TenantContext":
        if not isinstance(tenant_id, str) or not _TENANT_RE.fullmatch(tenant_id):
            raise InvalidInput("tenant id must be 2-63 characters of a-z, 0-9 and '-', starting with a letter or digit")
        return cls(tenant_id)

    def __str__(self) -> str:
        return self.tenant_id
