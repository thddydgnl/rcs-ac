"""Shared evaluation error types."""

from __future__ import annotations


class CleanEvalRuntimeFailure(RuntimeError):
    def __init__(self, failure_class: str, done_reason: str, message: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.done_reason = done_reason
