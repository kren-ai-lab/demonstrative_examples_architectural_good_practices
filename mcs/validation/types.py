from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Severity = Literal["ERROR", "WARNING", "INFO"]


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    path: Optional[str] = None
    hint: Optional[str] = None

    def short(self) -> str:
        p = f" (path={self.path})" if self.path else ""
        return f"[{self.severity}] {self.code}: {self.message}{p}"
