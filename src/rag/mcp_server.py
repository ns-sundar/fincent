"""Model Context Protocol (MCP) server that exposes the FAISS vector_db.

The server advertises a single tool -- ``rag_search`` by default -- that
any MCP-compatible client (Claude Desktop, Cursor, custom LangChain
MCP adapters, etc.) can call to query our persisted vector store at
``/data/vector_db``.

Run it as a standalone process:

.. code-block:: bash

    python -m src.rag.mcp_server

Transports:

* ``stdio`` (default)    -- suitable for client-spawned servers
  (Claude Desktop / Cursor register the command line and read/write on
  stdio).
* ``streamable-http``    -- long-lived HTTP server with bidirectional
  streaming on the ``/mcp`` endpoint. This is the replacement for the
  (now deprecated) SSE transport in the MCP spec / Python SDK.

The server is optional. The in-process Q&A agent always talks to the
same :func:`src.rag.tool.rag_search` function directly, so disabling
the MCP sidecar does not affect Q&A behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.config import AppConfig, get_config
from src.rag.tool import KNOWN_SOURCES, rag_search, to_wire_many
from src.utils.logging import configure_logging, get_logger

_logger = get_logger(__name__)


# Tool description visible to MCP clients in ``tools/list``.
_TOOL_DESCRIPTION = (
    "Search the Fincent FAISS vector_db (ingested from curated IRS / "
    "SEC / FINRA / FDIC / Fed / OCC / Treasury / Investopedia / ... "
    "sources) for passages relevant to QUERY. Optionally restrict hits "
    "to a single source via the SOURCE argument; supported values: "
    + ", ".join(KNOWN_SOURCES)
    + ". Each result carries title, url, tags (incl. source and "
    "category list), score, and the chunk text. Uses MMR by default "
    "for diversity; top_k defaults to cfg.rag.top_k."
)


def _run_search(
    query: str,
    source: Optional[str] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Thin wrapper so the MCP tool body stays trivial."""
    try:
        hits = rag_search(query, source=source, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 -- MCP tools must not crash the server
        _logger.exception("rag_search failed")
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    return to_wire_many(hits)


def build_server(cfg: Optional[AppConfig] = None) -> Any:
    """Construct a ``FastMCP`` server with the ``rag_search`` tool.

    Imports ``mcp`` lazily: the dependency is optional and only needed
    when the MCP sidecar is actually run. This keeps the rest of the
    application (Q&A agent, tests) free of the ``mcp`` install.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover -- exercised via README
        raise ImportError(
            "The 'mcp' package is required to run the MCP server. "
            "Install it with `pip install mcp>=1.2.0`."
        ) from exc

    cfg = cfg or get_config()
    tool_name = cfg.rag.mcp_server.tool_name

    server = FastMCP(
        name="fincent-rag",
        instructions=(
            "Retrieval tool over the Fincent curated finance corpus "
            "(IRS, SEC, FINRA, FDIC, Fed, OCC, Treasury, Investopedia, "
            "Bogleheads, etc.). Call it to ground answers about "
            "financial concepts; filter by source when the user "
            "requests it."
        ),
    )

    @server.tool(name=tool_name, description=_TOOL_DESCRIPTION)
    def _rag_search(
        query: str,
        source: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over the ingested FAISS vector store."""
        return _run_search(query, source=source, top_k=top_k)

    return server


def main() -> None:
    """Run the MCP server on the configured transport.

    The entrypoint is intentionally tiny: it defers every decision
    (transport, host, port) to ``cfg.rag.mcp_server`` so operators can
    reconfigure via ``config.yaml`` or ``FINCENT__RAG__MCP_SERVER__*``
    without editing code.
    """
    cfg = get_config()
    configure_logging(level=cfg.logging.level, log_file=cfg.logging.file or None)

    mcp_cfg = cfg.rag.mcp_server
    server = build_server(cfg)

    transport = (mcp_cfg.transport or "stdio").lower()
    _logger.info(
        "Starting Fincent MCP server (transport=%s, tool=%s)",
        transport,
        mcp_cfg.tool_name,
    )
    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "streamable-http":
        # FastMCP reads host/port from its settings object; pass them
        # through explicitly so ``config.yaml`` / env overrides work.
        server.settings.host = mcp_cfg.host
        server.settings.port = mcp_cfg.port
        server.run(transport="streamable-http")
    elif transport == "sse":  # pragma: no cover -- explicit deprecation path
        raise ValueError(
            "The 'sse' MCP transport is deprecated. "
            "Use 'streamable-http' (exposed on /mcp) or 'stdio'."
        )
    else:  # pragma: no cover -- guarded at config load time in future
        raise ValueError(
            f"Unsupported MCP transport '{transport}'. "
            "Use 'stdio' or 'streamable-http'."
        )


if __name__ == "__main__":  # pragma: no cover
    main()
