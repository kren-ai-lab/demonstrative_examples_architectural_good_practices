from __future__ import annotations

from importlib import metadata

_PACKAGE_NAME = "minimal-community-standard"

def get_mcs_version() -> str:
    """
    Return the installed package version as the default MCS schema version.

    Notes
    -----
    This ensures the MCS version is sourced from package metadata
    (pyproject.toml), avoiding hard-coded duplicates across schemas.
    """
    try:
        return metadata.version(_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        # Common during local dev when running without installation.
        # Keep deterministic fallback, but do not duplicate version elsewhere.
        return "0.0.0-dev"
