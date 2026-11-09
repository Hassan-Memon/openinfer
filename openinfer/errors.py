"""Exception hierarchy shared by backends, router and server.

The server turns any OpenInferError into an OpenAI style error body
(`{"error": {"message", "type", "code"}}`) with `status_code`. Backends raise BackendError
subclasses; the router looks at `retriable` to decide between "retry the same backend with
backoff" and "move on to the next candidate in the chain".
"""

from __future__ import annotations


class OpenInferError(Exception):
    status_code: int = 500
    error_type: str = "server_error"

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code

    def to_openai(self) -> dict[str, dict[str, str | None]]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.__class__.__name__,
                "param": None,
            }
        }


class BackendError(OpenInferError):
    """A backend failed to serve a request.

    `retriable=True` means the same backend may succeed if asked again shortly
    (rate limit, 5xx, timeout). `retriable=False` means try the next candidate.
    """

    status_code = 502
    error_type = "backend_error"

    def __init__(
        self,
        message: str,
        *,
        backend: str = "",
        status_code: int | None = None,
        retriable: bool = False,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.backend = backend
        self.retriable = retriable

    def __str__(self) -> str:
        prefix = f"[{self.backend}] " if self.backend else ""
        return f"{prefix}{self.message}"


class BackendUnavailable(BackendError):
    """Connection refused, DNS failure, backend disabled. Never retried on the same backend."""

    status_code = 503

    def __init__(self, message: str, *, backend: str = "") -> None:
        super().__init__(message, backend=backend, retriable=False)


class BackendTimeout(BackendError):
    status_code = 504

    def __init__(self, message: str, *, backend: str = "") -> None:
        super().__init__(message, backend=backend, retriable=True)


class RateLimited(BackendError):
    status_code = 429

    def __init__(self, message: str, *, backend: str = "") -> None:
        super().__init__(message, backend=backend, retriable=True)


class ModelNotFound(OpenInferError):
    status_code = 404
    error_type = "invalid_request_error"


class NoCandidates(OpenInferError):
    """Policy filters (privacy, cost, latency) removed every candidate. Fails closed."""

    status_code = 422
    error_type = "invalid_request_error"


class AllBackendsFailed(OpenInferError):
    status_code = 502
    error_type = "backend_error"

    def __init__(self, attempts: list[str]) -> None:
        self.attempts = attempts
        detail = "; ".join(attempts) if attempts else "no attempts were made"
        super().__init__(f"All candidates failed: {detail}")


class NotSupported(OpenInferError):
    status_code = 501
    error_type = "not_supported"


class AuthError(OpenInferError):
    status_code = 401
    error_type = "authentication_error"


class InvalidRequest(OpenInferError):
    status_code = 400
    error_type = "invalid_request_error"
