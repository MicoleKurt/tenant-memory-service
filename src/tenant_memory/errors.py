"""Exceptions. Each one maps to a distinct HTTP outcome in ``api.py``."""

from __future__ import annotations


class TenantMemoryError(Exception):
    """Base class for everything this package raises on purpose."""


class InvalidInput(TenantMemoryError):
    """The caller sent something that is not acceptable (HTTP 422)."""


class AuthError(TenantMemoryError):
    """Missing or unknown credentials (HTTP 401)."""


class Forbidden(TenantMemoryError):
    """Known credentials used on a route they may not call (HTTP 403)."""


class IsolationError(TenantMemoryError):
    """A result belonging to a different tenant reached the store boundary.

    This should be impossible, because every query carries a server-side tenant
    filter. It is checked anyway: if it ever fires, the request fails closed and
    nothing from the other tenant is returned.
    """


class AnonymizationError(InvalidInput):
    """Something that could identify a client was found where only structure is allowed."""


class EmbeddingError(TenantMemoryError):
    """The embedding provider failed or returned something unusable (HTTP 502)."""


class ConfigError(TenantMemoryError):
    """The service is misconfigured and must not start."""
