from __future__ import annotations

from .version import get_mcs_version
from mcs.api import (
    load_specs,
    validate_specs,
    create_record,
    register_run,
    run_pack,
)

__version__ = get_mcs_version()

__all__ = [
    "__version__", 
    "get_mcs_version",
    "load_specs",
    "validate_specs",
    "create_record",
    "register_run",
    "run_pack",]

