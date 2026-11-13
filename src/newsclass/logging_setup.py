"""Structured logging configuration.

Production logs should be machine-parsable and never leak the full request body
(news text can contain personal data). We log metadata only.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, in a format that works in containers."""
    numeric = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric)

    # These libraries are extremely chatty at INFO and add no operational value.
    for noisy in ("urllib3", "httpx", "httpcore", "filelock", "transformers", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
