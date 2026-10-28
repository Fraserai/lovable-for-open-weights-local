"""One error shape for every tool and endpoint: what happened, the exact fix, and whether to retry.

An agent reads these, so `fix` is a concrete next step ("Pass assistant_column='intent'"), never
"invalid input".
"""

from typing import Any


class WRError(Exception):
    def __init__(self, code: str, message: str, fix: str = "", *, retryable: bool = False, status: int = 400,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix = fix
        self.retryable = retryable
        self.status = status
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message, "fix": self.fix,
                               "retryable": self.retryable}
        if self.details:
            out["details"] = self.details
        return {"error": out}


def not_found(what: str, ident: str) -> WRError:
    return WRError(f"{what}_not_found", f"No {what.replace('_', ' ')} {ident!r} on this account.",
                   f"Check the id. list_jobs shows this account's jobs.", status=404)
