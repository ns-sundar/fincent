"""Document loaders for the RAG ingestion pipeline.

Responsibilities:
  * Download each URL from the article catalog.
  * Detect whether it is PDF or HTML from the Content-Type header
    (with a filename extension fallback).
  * Parse it using a LangChain loader -- ``UnstructuredPDFLoader`` for
    PDFs (unstructured.io under the hood) and ``BSHTMLLoader`` for
    HTML (BeautifulSoup under the hood).
  * Stamp the article's ``url``, ``title``, and ``tags`` onto every
    resulting ``Document.metadata`` so they flow through chunking and
    end up attached to each vector in FAISS.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse

import requests
from langchain_core.documents import Document

from src.utils.logging import get_logger

_logger = get_logger(__name__)


# Content-types that we treat as PDF. The server sometimes serves PDFs
# with a charset parameter, so we match on a prefix instead of equality.
_PDF_CT_PREFIXES: tuple[str, ...] = ("application/pdf", "application/x-pdf")


class ArticleFetcher(Protocol):
    """Interface used by tests to inject a fake HTTP client."""

    def __call__(
        self, url: str, *, timeout: int, user_agent: str
    ) -> "FetchedArticle": ...


class FetchedArticle:
    """Container for bytes + content-type returned by the fetcher."""

    __slots__ = ("content", "content_type", "final_url")

    def __init__(self, content: bytes, content_type: str, final_url: str) -> None:
        self.content: bytes = content
        self.content_type: str = content_type
        self.final_url: str = final_url


# HTTP statuses that a short retry can plausibly clear (edge rate limits,
# transient anti-bot challenges, upstream hiccups).
_RETRYABLE_STATUSES: frozenset[int] = frozenset({403, 408, 425, 429, 500, 502, 503, 504})


def _browser_headers(url: str, user_agent: str) -> Dict[str, str]:
    """Return a browser-class header set for ``url``.

    Many government and finance sites (investor.gov, cbp.gov, usitc.gov,
    investopedia.com, finra.org ...) return 403 to clients that look
    "non-browser" -- either because the UA is unknown or because
    ``Accept`` / ``Accept-Language`` / ``Sec-Fetch-*`` / ``Referer`` are
    missing. This header set mimics a vanilla Chrome navigation and
    clears the vast majority of those 403s.
    """
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
    headers: Dict[str, str] = {
        "User-Agent": user_agent,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "application/pdf;q=0.9,image/avif,image/webp,image/apng,"
            "*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        # Only advertise encodings ``requests`` decodes natively.
        # Brotli (``br``) would require the optional ``brotli`` package;
        # advertising it without that dep leaves caller receiving raw
        # brotli bytes that BSHTMLLoader cannot utf-8 decode.
        "Accept-Encoding": "gzip, deflate",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
        "Cache-Control": "max-age=0",
        "DNT": "1",
    }
    # A same-origin Referer convinces investopedia / investor.gov etc.
    # that we navigated here rather than scraping cold.
    if origin:
        headers["Referer"] = origin + "/"
    return headers


def default_fetcher(
    url: str, *, timeout: int, user_agent: str
) -> FetchedArticle:
    """Download ``url`` with ``requests`` and return bytes + content-type.

    Behaviour:
      * Sends a browser-class header set (see :func:`_browser_headers`)
        so that CDN/WAF bot filters on ``*.gov``, ``investopedia.com``,
        ``investor.gov`` etc. accept the request.
      * Follows redirects and honors the caller-supplied timeout.
      * Retries once, with a short backoff, on the subset of HTTP
        statuses in :data:`_RETRYABLE_STATUSES` -- this clears
        transient edge-rate-limits and bot challenges without masking
        genuine 404s.
      * Raises ``requests.RequestException`` / ``HTTPError`` on
        transport failures so the ingestion layer can log-and-skip per
        URL (the exception's status code is preserved on
        ``exc.response.status_code``).
    """
    headers = _browser_headers(url, user_agent)

    last_exc: Optional[requests.RequestException] = None
    for attempt in range(2):  # one initial attempt + one retry
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers=headers,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(1.0)
                continue
            raise

        if resp.status_code in _RETRYABLE_STATUSES and attempt == 0:
            _logger.debug(
                "Retrying %s after HTTP %s (attempt 2/2)", url, resp.status_code
            )
            time.sleep(1.5)
            continue

        resp.raise_for_status()
        content_type = (resp.headers.get("Content-Type") or "").lower()
        return FetchedArticle(
            content=resp.content,
            content_type=content_type,
            final_url=resp.url or url,
        )

    # Retry path exhausted without a usable response; surface the last
    # exception if we have one, otherwise re-raise on the final response.
    if last_exc is not None:
        raise last_exc
    # Defensive: the for-loop above always either returns or raises,
    # so this line is effectively unreachable.
    raise RuntimeError(f"Unexpected fetcher flow for {url}")


def _looks_like_pdf(content_type: str, url: str) -> bool:
    """Heuristic: PDF if the server says so, or the URL ends in ``.pdf``."""
    if any(content_type.startswith(p) for p in _PDF_CT_PREFIXES):
        return True
    # Some government sites serve PDFs as octet-stream or miss the header
    # entirely -- fall back to the URL suffix.
    return url.lower().split("?", 1)[0].endswith(".pdf")


def _load_pdf_bytes(pdf_bytes: bytes) -> List[Document]:
    """Parse PDF bytes into ``Document`` objects.

    Tries ``UnstructuredPDFLoader`` first (the user-requested path) and
    falls back to ``PyPDFLoader`` if unstructured fails or is missing
    its system-level dependencies (poppler, tesseract, etc.).
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        try:
            # Preferred path: unstructured.io via LangChain loader.
            from langchain_community.document_loaders import UnstructuredPDFLoader

            loader = UnstructuredPDFLoader(
                str(tmp_path),
                mode="single",
                strategy="fast",
            )
            return loader.load()
        except Exception as exc:  # noqa: BLE001 -- fall back below
            _logger.warning(
                "UnstructuredPDFLoader failed (%s); falling back to PyPDFLoader",
                exc,
            )
            from langchain_community.document_loaders import PyPDFLoader

            return PyPDFLoader(str(tmp_path)).load()
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _load_html_bytes(html_bytes: bytes) -> List[Document]:
    """Parse HTML bytes into ``Document`` objects via LangChain's BSHTMLLoader."""
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="wb"
    ) as tmp:
        tmp.write(html_bytes)
        tmp_path = Path(tmp.name)
    try:
        from langchain_community.document_loaders import BSHTMLLoader

        # ``bs_kwargs`` uses lxml for speed; ``open_encoding`` lets us
        # tolerate the mixed encodings real government pages ship with.
        loader = BSHTMLLoader(
            str(tmp_path),
            open_encoding="utf-8",
            bs_kwargs={"features": "lxml"},
        )
        return loader.load()
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _stamp_metadata(
    docs: List[Document],
    *,
    url: str,
    title: str,
    tags: Dict[str, Any],
) -> List[Document]:
    """Merge catalog metadata onto every loaded Document."""
    for doc in docs:
        meta = dict(doc.metadata or {})
        # Flatten/copy the catalog fields. ``tags`` is kept as a nested
        # dict because FAISS preserves arbitrary JSON-serialisable
        # metadata; downstream consumers can read categories off it.
        meta["url"] = url
        meta["title"] = title
        meta["tags"] = dict(tags) if tags else {}
        # Source is a convention LangChain retrievers rely on.
        meta.setdefault("source", url)
        doc.metadata = meta
    return docs


def load_article(
    article: Dict[str, Any],
    *,
    timeout: int,
    user_agent: str,
    fetcher: Optional[ArticleFetcher] = None,
) -> List[Document]:
    """Download and parse a single article catalog entry.

    Args:
        article: ``{"url": str, "title": str, "tags": {...}}``.
        timeout: Per-request HTTP timeout (seconds).
        user_agent: UA string for the download request.
        fetcher: Optional HTTP client override (tests inject fakes).

    Returns:
        A list of LangChain ``Document`` objects (typically one
        per source page) with catalog metadata attached. May be empty
        if the source yielded no extractable text.

    Raises:
        Any exception from the underlying fetcher/loader. The
        ingestion layer catches these per-URL.
    """
    url = article["url"]
    title = article.get("title", "")
    tags = article.get("tags", {}) or {}

    fetch = fetcher or default_fetcher
    fetched = fetch(url, timeout=timeout, user_agent=user_agent)

    if _looks_like_pdf(fetched.content_type, fetched.final_url):
        _logger.debug("Parsing %s as PDF", url)
        docs = _load_pdf_bytes(fetched.content)
    else:
        _logger.debug("Parsing %s as HTML", url)
        docs = _load_html_bytes(fetched.content)

    # Drop empty docs (some loaders return a placeholder with no text).
    docs = [d for d in docs if (d.page_content or "").strip()]
    return _stamp_metadata(docs, url=url, title=title, tags=tags)
