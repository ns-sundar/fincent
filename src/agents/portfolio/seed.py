"""One-shot seeding of the runtime portfolio directory.

The app reads the user's portfolio from ``cfg.portfolio.data_path`` (a
writable path on the host, e.g. ``/data/portfolio`` on HuggingFace
Spaces). On a fresh deployment that directory is empty, so on startup
we copy every JSON file from ``cfg.portfolio.seed_path`` (a read-only
snapshot shipped in the repo, e.g. ``data/default_portfolio``) into
``data_path`` -- but **only** for files that do not already exist
there.

The result:

* On **first** boot, ``/data/portfolio/`` is populated with the same
  ``accounts.json`` + ``transactions.json`` that the repo ships.
* On **subsequent** boots, any file already present in
  ``/data/portfolio/`` is left alone. This lets future UI / agent
  features persist user edits without every restart stomping them.
* The function is **idempotent**: calling it twice in a row is a
  no-op after the first call.

In addition to the live data files the seeder also writes
``sample-accounts.json`` and ``sample-transactions.json`` -- permanent,
read-only copies of the default portfolio -- so the Streamlit UI can
offer them as example downloads.  These sample files are only created
once and never overwritten.

The seeder is called from two places to cover every startup path:

1. The FastAPI lifespan (``src.workflow.server``) -- so logs surface
   any failure before the first request is served.
2. Lazily from ``src.agents.portfolio.loader`` -- so a Streamlit-only
   launch or a direct ``load_portfolio()`` call is still safe.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

from src.core.config import AppConfig, get_config
from src.utils.logging import get_logger

_logger = get_logger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# Source file names in seed_path → sample file names written to data_path.
_SAMPLE_MAP: dict[str, str] = {
    "accounts.json": "sample-accounts.json",
    "transactions.json": "sample-transactions.json",
}


def _resolve(path_str: str) -> Path:
    """Resolve a config path: absolute stays as-is, relative -> repo root."""
    p = Path(path_str)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _write_sample(src_file: Path, dst_file: Path) -> None:
    """Copy *src_file* to *dst_file* and make *dst_file* read-only."""
    shutil.copy2(src_file, dst_file)
    # Remove write bits for owner, group, and others.
    current = dst_file.stat().st_mode
    dst_file.chmod(current & ~0o222)


def seed_portfolio_if_needed(cfg: Optional[AppConfig] = None) -> Path:
    """Ensure ``cfg.portfolio.data_path`` is populated from ``seed_path``.

    Returns the resolved runtime ``data_path`` regardless of whether
    seeding actually copied anything (idempotent). If the target
    directory cannot be created -- most commonly because the host
    ``/data`` mount is missing in a dev environment -- the function
    logs a warning and returns the seed path instead so the caller
    can still read the data in read-only mode.
    """
    cfg = cfg or get_config()
    target = _resolve(cfg.portfolio.data_path)
    source = _resolve(cfg.portfolio.seed_path)

    if not source.is_dir():
        _logger.warning(
            "Portfolio seed_path %s is missing; cannot seed %s.",
            source,
            target,
        )
        return target

    if target.resolve() == source.resolve():
        # Nothing to do: the app is pointed straight at the read-only
        # seed (e.g. local dev override). The loader will read it in
        # place.
        return target

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _logger.warning(
            "Cannot create portfolio data_path %s (%s); "
            "falling back to read-only seed at %s.",
            target,
            exc,
            source,
        )
        return source

    # Copy only the two canonical live-data files (not schemas, alt-*, etc.)
    # and write the matching read-only sample file at the same time.
    _SEED_FILES = ("accounts.json", "transactions.json")
    copied: List[str] = []
    for src_file in [source / name for name in _SEED_FILES]:
        dst_file = target / src_file.name
        if not dst_file.exists():
            try:
                shutil.copy2(src_file, dst_file)
                copied.append(src_file.name)
            except OSError as exc:
                _logger.warning(
                    "Failed to seed %s -> %s: %s", src_file, dst_file, exc
                )

        # Write the read-only sample alongside the live file (once only).
        sample_name = _SAMPLE_MAP.get(src_file.name)
        if sample_name:
            sample_file = target / sample_name
            if not sample_file.exists():
                try:
                    _write_sample(src_file, sample_file)
                    _logger.info("Wrote read-only sample %s", sample_file)
                except OSError as exc:
                    _logger.warning("Failed to write sample %s: %s", sample_file, exc)

    if copied:
        _logger.info(
            "Seeded portfolio data_path=%s from seed_path=%s (copied=%s)",
            target,
            source,
            copied,
        )
    else:
        _logger.debug(
            "Portfolio data_path=%s already populated; nothing to seed.",
            target,
        )

    return target


__all__ = ["seed_portfolio_if_needed"]
