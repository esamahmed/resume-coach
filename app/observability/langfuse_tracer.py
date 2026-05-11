"""
app/observability/langfuse_tracer.py

Langfuse span-level tracing — every agent node emits its own span.
Falls back silently to no-op if Langfuse is disabled or keys are missing.
The pipeline NEVER fails due to tracing errors.

Compatible with Langfuse Python SDK v4.x.
Uses get_client() + start_as_current_observation() — the v4 API.

Set LANGFUSE_ENABLED=false in .env to skip tracing during local dev.
"""
from __future__ import annotations
import logging
import os
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
    def set_trace_io(self, **kwargs: Any) -> None:
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
            and not settings.langfuse_public_key.startswith("pk-lf-YOUR")
        )
        self._client = None

        if self._enabled:
            try:
                # Set env vars before get_client() — v4 reads from environment
                os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
                os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
                os.environ["LANGFUSE_HOST"] = settings.langfuse_host

                from langfuse import get_client
                self._client = get_client()
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
        Context manager that emits a Langfuse span using v4 API.
        Never raises — pipeline continues normally on tracing errors.
        """
        if not self._enabled or self._client is None:
            yield _NullSpan()
            return

        try:
            from langfuse import propagate_attributes

            with propagate_attributes(
                trace_name=f"resume-coach-{name}",
                session_id=session_id or None,
                tags=["resume-coach"],
            ):
                with self._client.start_as_current_observation(
                    as_type="span",
                    name=name,
                    input=input_data or {},
                ) as span:
                    try:
                        yield span
                    except Exception as exc:
                        try:
                            span.update(output={"error": str(exc)})
                        except Exception:
                            pass
                        raise

        except Exception as e:
            logger.warning("Langfuse span failed, continuing: %s", e)
            yield _NullSpan()

    def get_trace_url(self, session_id: str) -> str:
        if not self._enabled or self._client is None:
            return ""
        settings = get_settings()
        return f"{settings.langfuse_host}/sessions/{session_id}"

    def flush(self) -> None:
        if self._enabled and self._client:
            try:
                self._client.flush()
                logger.debug("Langfuse flush complete")
            except Exception as e:
                logger.warning("Langfuse flush failed: %s", e)


_tracer: LangfuseTracer | None = None


def get_tracer() -> LangfuseTracer:
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer
