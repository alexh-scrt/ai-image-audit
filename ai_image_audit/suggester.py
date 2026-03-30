"""Openverse API suggester for the AI Image Audit tool.

This module queries the `Openverse <https://openverse.org>`_ open image
search API to find Creative Commons licensed, human-made alternative images
for a given search query.  Openverse provides access to hundreds of millions
of openly licensed images sourced from cultural institutions and open
collections around the world.

The primary entry point is :func:`fetch_suggestions`, which accepts a text
query (typically derived from image metadata, alt-text, or inferred tags)
and returns a list of :class:`ImageSuggestion` dataclass instances.

Openverse API documentation: https://api.openverse.org/v1/

Typical usage::

    from ai_image_audit.suggester import fetch_suggestions

    suggestions = fetch_suggestions("forest landscape photography", per_page=5)
    for s in suggestions:
        print(s.title, s.url, s.thumbnail)

Or with a pre-configured :class:`OpenverseSuggester` instance::

    from ai_image_audit.suggester import OpenverseSuggester

    suggester = OpenverseSuggester(per_page=5)
    suggestions = suggester.suggest("mountain sunset")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Base URL for the Openverse v1 API.
OPENVERSE_API_BASE: str = "https://api.openverse.org/v1"

#: Endpoint path for image search.
_IMAGE_SEARCH_PATH: str = "/images/"

#: Default number of suggestions to request per query.
DEFAULT_PER_PAGE: int = 5

#: Maximum number of suggestions the API will allow per request.
_MAX_PER_PAGE: int = 20

#: Default HTTP request timeout in seconds.
DEFAULT_TIMEOUT: int = 10

#: User-Agent header sent with Openverse API requests.
_USER_AGENT: str = (
    "AIImageAuditBot/0.1 (https://github.com/ai-image-audit; "
    "contact@example.com)"
)

#: Licence filter – only return images with these Creative Commons licences.
#: These correspond to the most permissive CC licences suitable for most
#: commercial and non-commercial projects.
_DEFAULT_LICENCE_FILTER: str = "cc0,by,by-sa"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ImageSuggestion:
    """A single royalty-free image suggestion returned by the Openverse API.

    Attributes:
        id: Openverse unique identifier for the image.
        title: Human-readable title of the image, if available.
        url: Direct URL to the full-resolution image file.
        thumbnail: URL to a smaller thumbnail version of the image, if
            available.
        foreign_landing_url: URL to the image's page on the originating
            provider website (e.g. Flickr, Wikimedia Commons).
        creator: Name of the image creator/author, if known.
        creator_url: URL to the creator's profile page, if available.
        licence: Short licence identifier string (e.g. ``"cc0"``,
            ``"by"``, ``"by-sa"``).
        licence_version: Version of the licence (e.g. ``"4.0"``).
        licence_url: URL to the full licence text.
        provider: Identifier of the upstream image provider
            (e.g. ``"flickr"``, ``"wikimedia_commons"``).
        source: Data source within the provider.
        tags: List of descriptive tags associated with the image.
        width: Image width in pixels, if available.
        height: Image height in pixels, if available.
    """

    id: str
    title: Optional[str]
    url: str
    thumbnail: Optional[str]
    foreign_landing_url: Optional[str]
    creator: Optional[str]
    creator_url: Optional[str]
    licence: str
    licence_version: Optional[str]
    licence_url: Optional[str]
    provider: Optional[str]
    source: Optional[str]
    tags: List[str] = field(default_factory=list)
    width: Optional[int] = field(default=None)
    height: Optional[int] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the suggestion to a JSON-compatible dictionary.

        Returns:
            A flat dictionary suitable for JSON serialisation.
        """
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "thumbnail": self.thumbnail,
            "foreign_landing_url": self.foreign_landing_url,
            "creator": self.creator,
            "creator_url": self.creator_url,
            "licence": self.licence,
            "licence_version": self.licence_version,
            "licence_url": self.licence_url,
            "provider": self.provider,
            "source": self.source,
            "tags": self.tags,
            "width": self.width,
            "height": self.height,
        }


# ---------------------------------------------------------------------------
# Suggester class
# ---------------------------------------------------------------------------


class OpenverseSuggester:
    """Client for the Openverse image search API.

    Manages HTTP session lifecycle and provides a simple
    :meth:`suggest` method that returns a list of
    :class:`ImageSuggestion` instances for a given query string.

    Args:
        per_page: Default number of suggestions to return per call.
            Capped at :data:`_MAX_PER_PAGE`.
        timeout: HTTP request timeout in seconds.
        licence_filter: Comma-separated list of CC licence identifiers
            to restrict results to.  Defaults to
            :data:`_DEFAULT_LICENCE_FILTER`.
        api_base: Base URL for the Openverse API.  Override for testing
            or if the API URL changes.
        session: Optional :class:`requests.Session` to reuse.  A new
            session is created if ``None``.
    """

    def __init__(
        self,
        per_page: int = DEFAULT_PER_PAGE,
        timeout: int = DEFAULT_TIMEOUT,
        licence_filter: str = _DEFAULT_LICENCE_FILTER,
        api_base: str = OPENVERSE_API_BASE,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Initialise the suggester.

        Args:
            per_page: Number of suggestions per query (1–20).
            timeout: Request timeout in seconds.
            licence_filter: Comma-separated CC licence filter string.
            api_base: Base URL for the Openverse API.
            session: Optional requests session.

        Raises:
            ValueError: If *per_page* is less than 1.
        """
        if per_page < 1:
            raise ValueError(f"per_page must be >= 1, got {per_page!r}.")

        self.per_page = min(per_page, _MAX_PER_PAGE)
        self.timeout = timeout
        self.licence_filter = licence_filter
        self.api_base = api_base.rstrip("/")

        self._own_session = session is None
        self._session: requests.Session = session or requests.Session()
        self._session.headers.update({
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        })

        logger.info(
            "OpenverseSuggester initialised: per_page=%d, api_base=%s",
            self.per_page,
            self.api_base,
        )

    def suggest(
        self,
        query: str,
        *,
        per_page: Optional[int] = None,
    ) -> List[ImageSuggestion]:
        """Fetch royalty-free image suggestions from the Openverse API.

        Args:
            query: Search query string.  Should be descriptive enough to
                return relevant Creative Commons licensed images (e.g.
                ``"mountain lake photography"`` rather than a single word).
            per_page: Number of results to return for this specific call.
                Overrides the instance-level default if provided.  Capped
                at :data:`_MAX_PER_PAGE`.

        Returns:
            A list of :class:`ImageSuggestion` instances.  May be empty
            if no results are found or the query is empty.

        Raises:
            requests.exceptions.RequestException: If the API call fails
                due to a network error or an HTTP error response.
        """
        if not query or not query.strip():
            logger.warning("suggest() called with empty query; returning empty list.")
            return []

        count = min(per_page or self.per_page, _MAX_PER_PAGE)
        url = self._build_url(query.strip(), count)

        logger.info("Fetching Openverse suggestions: query=%r, per_page=%d", query, count)

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("Openverse API request failed: %s", exc)
            raise

        try:
            data = response.json()
        except ValueError as exc:
            raise requests.exceptions.RequestException(
                f"Openverse API returned non-JSON response: {exc}"
            ) from exc

        results = data.get("results", [])
        if not isinstance(results, list):
            logger.warning(
                "Unexpected Openverse API response structure; 'results' is not a list."
            )
            return []

        suggestions = []
        for item in results:
            suggestion = _parse_suggestion(item)
            if suggestion is not None:
                suggestions.append(suggestion)

        logger.info(
            "Openverse returned %d suggestion(s) for query %r",
            len(suggestions),
            query,
        )
        return suggestions

    def close(self) -> None:
        """Close the underlying HTTP session if it was created internally.

        Safe to call multiple times.  Has no effect if the session was
        supplied externally.
        """
        if self._own_session:
            self._session.close()
            logger.debug("OpenverseSuggester session closed.")

    def __enter__(self) -> "OpenverseSuggester":
        """Support use as a context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the session on context manager exit."""
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_url(self, query: str, per_page: int) -> str:
        """Construct the full Openverse API search URL.

        Args:
            query: The search query string.
            per_page: Number of results to request.

        Returns:
            A fully qualified URL string with query parameters.
        """
        params: Dict[str, Any] = {
            "q": query,
            "page_size": per_page,
            "license": self.licence_filter,
            "mature": "false",
        }
        return f"{self.api_base}{_IMAGE_SEARCH_PATH}?{urlencode(params)}"


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def fetch_suggestions(
    query: str,
    *,
    per_page: int = DEFAULT_PER_PAGE,
    timeout: int = DEFAULT_TIMEOUT,
    licence_filter: str = _DEFAULT_LICENCE_FILTER,
    api_base: str = OPENVERSE_API_BASE,
    session: Optional[requests.Session] = None,
) -> List[ImageSuggestion]:
    """Convenience function to fetch Openverse image suggestions.

    Creates a temporary :class:`OpenverseSuggester` instance, calls
    :meth:`~OpenverseSuggester.suggest`, and returns the results.  For
    repeated queries, prefer instantiating :class:`OpenverseSuggester`
    once to reuse the HTTP session.

    Args:
        query: Search query string.
        per_page: Number of suggestions to return (1–20).
        timeout: HTTP request timeout in seconds.
        licence_filter: Comma-separated CC licence filter.
        api_base: Base URL for the Openverse API.
        session: Optional :class:`requests.Session` to use.

    Returns:
        A list of :class:`ImageSuggestion` instances.

    Raises:
        requests.exceptions.RequestException: On network or HTTP errors.
    """
    with OpenverseSuggester(
        per_page=per_page,
        timeout=timeout,
        licence_filter=licence_filter,
        api_base=api_base,
        session=session,
    ) as suggester:
        return suggester.suggest(query)


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------


def _parse_suggestion(item: Dict[str, Any]) -> Optional[ImageSuggestion]:
    """Parse a single Openverse API result item into an :class:`ImageSuggestion`.

    Silently returns ``None`` for items that are missing required fields
    (``id`` or ``url``) so that a single malformed result does not abort
    the entire response.

    Args:
        item: A dictionary representing a single image result from the
            Openverse ``/v1/images/`` endpoint.

    Returns:
        An :class:`ImageSuggestion` on success, or ``None`` if the item
        is missing required fields.
    """
    image_id = item.get("id")
    url = item.get("url")

    if not image_id or not url:
        logger.debug(
            "Skipping Openverse result missing 'id' or 'url': %s",
            {k: item.get(k) for k in ("id", "url", "title")},
        )
        return None

    # Extract tags – the API returns a list of dicts with a 'name' key.
    raw_tags = item.get("tags") or []
    tags: List[str] = []
    for tag in raw_tags:
        if isinstance(tag, dict):
            name = tag.get("name")
            if name and isinstance(name, str):
                tags.append(name)
        elif isinstance(tag, str):
            tags.append(tag)

    # Parse width and height safely.
    width: Optional[int] = None
    height: Optional[int] = None
    try:
        raw_width = item.get("width")
        if raw_width is not None:
            width = int(raw_width)
    except (ValueError, TypeError):
        pass
    try:
        raw_height = item.get("height")
        if raw_height is not None:
            height = int(raw_height)
    except (ValueError, TypeError):
        pass

    return ImageSuggestion(
        id=str(image_id),
        title=item.get("title") or None,
        url=str(url),
        thumbnail=item.get("thumbnail") or None,
        foreign_landing_url=item.get("foreign_landing_url") or None,
        creator=item.get("creator") or None,
        creator_url=item.get("creator_url") or None,
        licence=str(item.get("license") or ""),
        licence_version=item.get("license_version") or None,
        licence_url=item.get("license_url") or None,
        provider=item.get("provider") or None,
        source=item.get("source") or None,
        tags=tags,
        width=width,
        height=height,
    )
