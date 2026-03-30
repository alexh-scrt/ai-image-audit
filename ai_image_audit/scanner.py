"""Dual-mode image scanner for the AI Image Audit tool.

This module provides functionality to discover image assets from two sources:

1. **Local directory** – recursively walks a filesystem directory and returns
   the absolute paths of all image files whose extensions match a supported
   set (JPEG, PNG, GIF, WebP, BMP, TIFF, SVG).

2. **Website URL** – fetches the page at the given URL, parses the HTML with
   BeautifulSoup4, resolves all ``<img src>`` and ``<source srcset>`` image
   references to absolute URLs, and returns the de-duplicated list.

Both modes return a list of :class:`ImageRef` dataclass instances that
normalise the source information so downstream components can handle both
local and remote images uniformly.

Typical usage::

    from ai_image_audit.scanner import scan

    # Local directory
    refs = scan("/path/to/project/assets")

    # Remote website
    refs = scan("https://example.com")

    for ref in refs:
        print(ref.source, ref.is_remote)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: File extensions considered as images during local directory scans.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        ".svg",
    }
)

#: Default request timeout in seconds for URL-based scanning.
DEFAULT_REQUEST_TIMEOUT: int = 15

#: Default maximum number of images to return from a single scan.
DEFAULT_MAX_IMAGES: int = 200

#: User-Agent header sent when crawling remote pages.
_USER_AGENT: str = (
    "Mozilla/5.0 (compatible; AIImageAuditBot/0.1; +https://github.com/ai-image-audit)"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ImageRef:
    """A normalised reference to a single image asset.

    Attributes:
        source: For local images this is the absolute filesystem path as a
            string; for remote images it is the fully-qualified URL.
        is_remote: ``True`` when the image was discovered via a URL scan,
            ``False`` for local filesystem scans.
        origin: The root directory (local) or base URL (remote) from which
            this reference was discovered.  Useful for display purposes.
        alt_text: The ``alt`` attribute of the originating ``<img>`` tag, if
            available.  Always ``None`` for local images.
        extension: Lowercase file extension including the leading dot, e.g.
            ``".jpg"``.  May be an empty string if the URL has no path
            extension.
    """

    source: str
    is_remote: bool
    origin: str
    alt_text: Optional[str] = field(default=None)
    extension: str = field(default="")

    def __post_init__(self) -> None:
        """Derive the extension from *source* if it was not supplied."""
        if not self.extension:
            if self.is_remote:
                path = urlparse(self.source).path
            else:
                path = self.source
            self.extension = Path(path).suffix.lower()

    def __repr__(self) -> str:  # pragma: no cover
        kind = "remote" if self.is_remote else "local"
        return f"ImageRef({kind}, {self.source!r})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(
    target: str,
    *,
    max_images: int = DEFAULT_MAX_IMAGES,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> List[ImageRef]:
    """Discover image assets from a local directory path or a website URL.

    The function auto-detects the mode by checking whether *target* looks like
    a URL (starts with ``http://`` or ``https://``) or a filesystem path.

    Args:
        target: Either an absolute/relative local directory path or a fully
            qualified HTTP/HTTPS URL to crawl.
        max_images: Upper bound on the number of :class:`ImageRef` objects
            returned.  Defaults to :data:`DEFAULT_MAX_IMAGES`.
        request_timeout: Timeout in seconds used for all outbound HTTP
            requests during URL-based scanning.  Ignored for local scans.
        session: Optional :class:`requests.Session` instance to reuse for
            all HTTP requests.  When ``None`` a temporary session is created
            internally.  Useful for testing (allows injection of mocked
            sessions).

    Returns:
        A list of :class:`ImageRef` instances, capped at *max_images*.

    Raises:
        ValueError: If *target* is an empty string.
        FileNotFoundError: If *target* is a local path that does not exist.
        NotADirectoryError: If *target* is a local path that points to a file
            rather than a directory.
        requests.exceptions.RequestException: If the initial HTTP fetch for a
            URL target fails (e.g. network error, 4xx/5xx response).
    """
    if not target or not target.strip():
        raise ValueError("scan() requires a non-empty target string.")

    target = target.strip()

    if _is_url(target):
        logger.info("Starting URL scan for: %s", target)
        return scan_url(
            target,
            max_images=max_images,
            request_timeout=request_timeout,
            session=session,
        )
    else:
        logger.info("Starting local directory scan for: %s", target)
        return scan_directory(target, max_images=max_images)


def scan_directory(
    directory: str,
    *,
    max_images: int = DEFAULT_MAX_IMAGES,
) -> List[ImageRef]:
    """Recursively collect image files from a local directory.

    Only files whose lowercase extension is present in
    :data:`SUPPORTED_EXTENSIONS` are included.  Symbolic links are followed.
    Hidden directories (names starting with ``'.'``) and common
    non-asset directories (``node_modules``, ``.git``, ``__pycache__``,
    ``venv``, ``.venv``) are skipped to avoid scanning dependency trees.

    Args:
        directory: Path to the root directory to scan.  May be relative; it
            will be resolved to an absolute path internally.
        max_images: Maximum number of results to return.

    Returns:
        A list of :class:`ImageRef` instances for each discovered image file,
        sorted by absolute path for deterministic output.

    Raises:
        FileNotFoundError: If *directory* does not exist.
        NotADirectoryError: If *directory* is not a directory.
    """
    root = Path(directory).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory!r}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory!r}")

    skip_dirs: frozenset[str] = frozenset(
        {
            "node_modules",
            ".git",
            "__pycache__",
            "venv",
            ".venv",
            ".tox",
            "dist",
            "build",
            ".mypy_cache",
            ".pytest_cache",
        }
    )

    refs: list[ImageRef] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        # Prune unwanted subtrees in-place so os.walk skips them entirely.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in skip_dirs and not d.startswith(".")
        ]
        dirnames.sort()  # Deterministic traversal order.

        for filename in sorted(filenames):
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            abs_path = str(Path(dirpath) / filename)
            refs.append(
                ImageRef(
                    source=abs_path,
                    is_remote=False,
                    origin=str(root),
                    alt_text=None,
                    extension=ext,
                )
            )

            if len(refs) >= max_images:
                logger.warning(
                    "scan_directory: reached max_images limit (%d) at %s",
                    max_images,
                    dirpath,
                )
                return refs

    logger.info(
        "scan_directory: found %d image(s) under %s", len(refs), root
    )
    return refs


def scan_url(
    url: str,
    *,
    max_images: int = DEFAULT_MAX_IMAGES,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> List[ImageRef]:
    """Crawl a single web page and collect all referenced image URLs.

    The function fetches *url*, parses the HTML, and extracts image
    references from the following sources:

    * ``<img src="…">`` attributes
    * ``<img srcset="…">`` attributes (all candidate URLs)
    * ``<source srcset="…">`` inside ``<picture>`` elements
    * ``<meta property="og:image" content="…">`` Open Graph tags

    All relative URLs are resolved against the base URL of the page.
    Duplicate URLs are removed (first occurrence wins) and the result is
    capped at *max_images*.

    Only URLs whose path ends with a supported image extension (or that have
    no recognisable extension but originate from an ``<img>`` tag) are
    returned; non-image links are silently dropped.

    Args:
        url: The fully qualified HTTP/HTTPS URL of the page to crawl.
        max_images: Maximum number of results to return.
        request_timeout: Timeout in seconds for the HTTP GET request.
        session: Optional :class:`requests.Session` to use.  A new session
            is created internally when ``None``.

    Returns:
        A de-duplicated list of :class:`ImageRef` instances.

    Raises:
        ValueError: If *url* is not a valid HTTP/HTTPS URL.
        requests.exceptions.RequestException: On network or HTTP errors.
    """
    if not _is_url(url):
        raise ValueError(f"scan_url() requires an http/https URL, got: {url!r}")

    own_session = session is None
    if own_session:
        session = requests.Session()
        session.headers["User-Agent"] = _USER_AGENT

    try:
        response = session.get(url, timeout=request_timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error("scan_url: failed to fetch %s – %s", url, exc)
        raise
    finally:
        if own_session:
            session.close()

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        logger.warning(
            "scan_url: %s returned non-HTML content-type %r; "
            "attempting parse anyway.",
            url,
            content_type,
        )

    base_url = _derive_base_url(url, response)
    soup = BeautifulSoup(response.text, "html.parser")

    seen: set[str] = set()
    refs: list[ImageRef] = []

    def _add_ref(src: str, alt: Optional[str] = None) -> bool:
        """Normalise *src*, deduplicate, and append to *refs*.

        Returns ``True`` when the image was added, ``False`` when skipped.
        """
        if not src or not src.strip():
            return False
        src = src.strip()

        # Resolve relative URLs.
        absolute = urljoin(base_url, src)

        # Validate the resolved URL is still http/https.
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            return False

        # Normalise by removing fragment identifiers.
        absolute = urlunparse(parsed._replace(fragment=""))

        if absolute in seen:
            return False

        ext = Path(urlparse(absolute).path).suffix.lower()

        # Allow images with a supported extension OR with no extension
        # (e.g. dynamic image endpoints like /api/image?id=123) when they
        # come from an <img> tag.
        if ext and ext not in SUPPORTED_EXTENSIONS:
            return False

        seen.add(absolute)
        refs.append(
            ImageRef(
                source=absolute,
                is_remote=True,
                origin=url,
                alt_text=alt or None,
                extension=ext,
            )
        )
        return True

    # --- <img> tags ---
    for img in soup.find_all("img"):
        if len(refs) >= max_images:
            break
        alt = img.get("alt", "") or ""
        src = img.get("src", "")
        if src:
            _add_ref(src, alt)
        # Also handle srcset on <img>.
        srcset = img.get("srcset", "")
        if srcset:
            for candidate_url in _parse_srcset(srcset):
                if len(refs) >= max_images:
                    break
                _add_ref(candidate_url, alt)

    # --- <source> tags inside <picture> ---
    for source_tag in soup.find_all("source"):
        if len(refs) >= max_images:
            break
        srcset = source_tag.get("srcset", "")
        if srcset:
            for candidate_url in _parse_srcset(srcset):
                if len(refs) >= max_images:
                    break
                _add_ref(candidate_url)

    # --- Open Graph image meta tags ---
    for meta in soup.find_all("meta", attrs={"property": re.compile(r"og:image", re.I)}):
        if len(refs) >= max_images:
            break
        content = meta.get("content", "")
        if content:
            _add_ref(content)

    logger.info(
        "scan_url: found %d image reference(s) on %s", len(refs), url
    )
    return refs


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_url(target: str) -> bool:
    """Return ``True`` if *target* looks like an HTTP or HTTPS URL.

    Args:
        target: The string to test.

    Returns:
        ``True`` when *target* starts with ``http://`` or ``https://``
        (case-insensitive), ``False`` otherwise.
    """
    lower = target.lower()
    return lower.startswith("http://") or lower.startswith("https://")


def _derive_base_url(requested_url: str, response: requests.Response) -> str:
    """Determine the effective base URL after any redirects.

    Uses the final URL from the response history (after redirects) as the
    base for resolving relative image references.

    Args:
        requested_url: The URL originally requested by the caller.
        response: The :class:`requests.Response` object from the GET request.

    Returns:
        The effective base URL string.
    """
    return response.url if response.url else requested_url


def _parse_srcset(srcset: str) -> list[str]:
    """Extract all URL candidates from an HTML ``srcset`` attribute string.

    The ``srcset`` format is a comma-separated list of image candidate
    strings where each entry is ``<url> [descriptor]``.  For example::

        "image-300.jpg 300w, image-600.jpg 600w, image-900.jpg 900w"

    Args:
        srcset: Raw value of a ``srcset`` attribute.

    Returns:
        A list of URL strings extracted from the srcset value.  May be
        empty if the input is blank or malformed.
    """
    urls: list[str] = []
    if not srcset:
        return urls

    # Each candidate is separated by a comma, but commas can appear inside
    # data: URIs.  We use a simple heuristic: split on ", " patterns where
    # the part before the comma is not a bare URL fragment.
    # The standard approach: split on commas, then take the first
    # whitespace-delimited token from each candidate part.
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        # The URL is the first whitespace-separated token.
        url = part.split()[0] if part.split() else ""
        if url:
            urls.append(url)

    return urls
