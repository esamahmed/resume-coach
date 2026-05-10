"""
app/observability/langfuse_tracer.py

Langfuse span-level tracing — every agent node emits its own span.
Falls back silently to no-op if Langfuse is disabled, keys are missing,
or the API version differs. The pipeline NEVER fails due to tracing errors.

Compatible with Langfuse v4.x SDK.
Set LANGFUSE_ENABLED=false in .env to skip tracing entirely during local dev.
"""
from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import Any, Generator

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class _NullSpan:
    """No-op span — used when Langfuse is disabled or keys are missing."""
    def update(self, **kwargs: Any) -> None:
        pass
    def end(self) -> None:
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class LangfuseTracer:
    def __init__(self):
        settings = get_settings()
        self._enabled = (
            settings.langfuse_enabled
            and bool(settings.langfuse_public_key)
            and bool(settings.langfuse_secret_key)
        )
        self._client = None

        if self._enabled:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                logger.info("Langfuse tracer initialised — host=%s", settings.langfuse_host)
            except Exception as e:
                logger.warning("Langfuse init failed, using no-op tracer: %s", e)
                self._enabled = False

    @contextmanager
    def span(
        self,
        name: str,
        session_id: str = "",
        input_data: dict | None = None,
        **kwargs: Any,
    ) -> Generator[Any, None, None]:
        """
        Context manager that emits a Langfuse span.
        Never raises — if tracing fails the pipeline continues normally.
        Compatible with Langfuse v4.x SDK.
        """
        if not self._enabled or self._client is None:
            yield _NullSpan()
            return

        trace = None
        span = None
        try:
            # Langfuse v4 API — create trace then span
            trace = self._client.trace(
                name=f"resume-coach-{name}",
                session_id=session_id or None,
                input=input_data or {},
                tags=["resume-coach"],
            )
            span = trace.span(
                name=name,
                input=input_data or {},
            )
        except Exception as e:
            logger.warning("Langfuse span creation failed, continuing: %s", e)
            yield _NullSpan()
            return

        try:
            yield span
        except Exception as exc:
            try:
                if span:
                    span.update(
                        metadata={"error": str(exc)},
                        level="ERROR",
                    )
            except Exception:
                pass
            raise
        finally:
            try:
                if span:
                    span.end()
            except Exception:
                pass
            try:
                if trace:
                    trace.update(output={"status": "complete"})
            except Exception:
                pass

    def get_trace_url(self, session_id: str) -> str:
        if not self._enabled or self._client is None:
            return ""
        settings = get_settings()
        return f"{settings.langfuse_host}/sessions/{session_id}"

    def flush(self) -> None:
        if self._enabled and self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning("Langfuse flush failed: %s", e)


_tracer: LangfuseTracer | None = None


def get_tracer() -> LangfuseTracer:
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer
