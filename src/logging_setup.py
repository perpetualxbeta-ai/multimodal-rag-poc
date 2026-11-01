"""Central logging configuration. Call ``setup_logging()`` once at program start."""

from __future__ import annotations

import logging
import os

_NOISY = ("httpx", "httpcore", "LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "urllib3", "PIL", "filelock")


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logging with timestamps and quiet down chatty third-party libraries.

    Third-party loggers stay at WARNING unless ``level`` is DEBUG.
    """
    numeric = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    third_party = logging.DEBUG if numeric <= logging.DEBUG else logging.WARNING
    for name in _NOISY:
        logging.getLogger(name).setLevel(third_party)
    os.environ.setdefault("LITELLM_LOG", "ERROR")  # litellm's own verbose printing
    try:
        import litellm

        litellm.suppress_debug_info = True  # hide the "Give Feedback / Get Help" banner on errors
    except ImportError:
        pass
