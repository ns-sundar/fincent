#!/usr/bin/env python3
"""Build the Fincent RAG FAISS index outside of the FastAPI server.

Useful when the host running the server cannot do large HTTPS uploads
(e.g. WSL2 with a middlebox that corrupts TLS records over ~64 KB).
Run this on any machine that CAN reach api.openai.com reliably, then
``scp`` the output directory onto the server host at the path your
``config.yaml`` (or ``FINCENT__RAG__VECTOR_DB_PATH``) points to.

Typical usage:

    # On a clean-network machine (Colab, a VPS, your phone's hotspot...)
    export OPENAI_API_KEY=sk-...
    python scripts/build_rag_index.py \
        --catalog rag/fincent_rag_articles.json \
        --out     ./vector_db_out

    # Then on the server host:
    sudo rm -rf /data/vector_db
    sudo mkdir -p /data
    scp -r user@build-host:/path/to/vector_db_out /data/vector_db
    ls /data/vector_db   # should list index.faiss and index.pkl

    # Start the server; ingestion will report state=skipped.
    ./run_local.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_config, reset_config_cache  # noqa: E402
from src.rag.ingest import ingest_if_needed  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--catalog",
        default=str(PROJECT_ROOT / "rag" / "fincent_rag_articles.json"),
        help="Path to the JSON article catalog (default: rag/fincent_rag_articles.json).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output directory for the FAISS index (will contain index.faiss + index.pkl).",
    )
    p.add_argument(
        "--level",
        default="INFO",
        help="Log level (DEBUG/INFO/WARNING/ERROR). Default: INFO.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    configure_logging(level=args.level)
    log = get_logger("build_rag_index")

    # Point the config at the requested catalog + output dir via env
    # overrides so we don't have to thread kwargs through the pipeline.
    import os

    os.environ["FINCENT__RAG__ENABLED"] = "true"
    os.environ["FINCENT__RAG__ARTICLES_PATH"] = str(Path(args.catalog).resolve())
    os.environ["FINCENT__RAG__VECTOR_DB_PATH"] = str(Path(args.out).resolve())
    reset_config_cache()

    cfg = get_config()
    log.info("Catalog: %s", cfg.rag.articles_path)
    log.info("Output : %s", cfg.rag.vector_db_path)

    snapshot = ingest_if_needed(cfg)
    log.info("Finished: state=%s detail=%s", snapshot.state, snapshot.detail)

    if snapshot.state in {"ready", "skipped"}:
        return 0
    log.error("Ingestion did not complete cleanly: %s", snapshot.error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
