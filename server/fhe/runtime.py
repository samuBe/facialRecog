"""Runtime checks and process-level compatibility guards for FHE backends."""

from __future__ import annotations

import os

_toy_fhe_activated = False
_native_fhe_initialized = False


def native_fhe_unavailable_reason() -> str | None:
    """Return a human-readable reason native FHE should stay disabled."""
    if os.getenv("FACIALRECOG_DISABLE_FHE", "").lower() in {"1", "true", "yes", "on"}:
        return "disabled by FACIALRECOG_DISABLE_FHE"
    if _toy_fhe_activated and not _native_fhe_initialized:
        return (
            "native FHE unavailable because toy OpenFHE mode was activated first in "
            "this process; HEIR setup crashes if openfhe-python loads before ctx.setup()"
        )
    return None


def native_fhe_enabled() -> bool:
    return native_fhe_unavailable_reason() is None


def claim_native_fhe() -> str | None:
    """Return a conflict reason if native HEIR setup is unsafe to start now."""
    reason = native_fhe_unavailable_reason()
    return reason


def toy_fhe_unavailable_reason() -> str | None:
    return None


def claim_toy_fhe() -> str | None:
    """Mark that the toy OpenFHE path has been activated in this process."""
    global _toy_fhe_activated
    reason = toy_fhe_unavailable_reason()
    if reason is not None:
        return reason
    _toy_fhe_activated = True
    return None


def mark_native_fhe_initialized() -> None:
    """Record that HEIR setup completed before any toy OpenFHE use."""
    global _native_fhe_initialized
    _native_fhe_initialized = True
