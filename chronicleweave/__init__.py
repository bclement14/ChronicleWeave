# chronicleweave/__init__.py
"""ChronicleWeave: multi-track Discord recording → readable script pipeline.

Public API is exposed lazily via PEP 562 ``__getattr__`` so importing the
package does not pull in numpy, pydub, google-genai, etc. unless the consumer
actually uses ``run_pipeline``. This matters for tools that only want, say,
``chronicleweave.modules.prepare_tracks``.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["run_pipeline"]

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import run_pipeline  # noqa: F401


def __getattr__(name: str) -> Any:
    if name == "run_pipeline":
        from .pipeline import run_pipeline as _rp
        return _rp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
