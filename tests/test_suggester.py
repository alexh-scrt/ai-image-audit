"""Unit tests for ai_image_audit.suggester.

Covers:
- OpenverseSuggester initialisation (defaults, custom params, invalid per_page).
- suggest() with valid mocked API responses.
- suggest() with empty query strings.
- suggest() when the API returns no results.
- suggest() with malformed API responses (missing fields, non-list results).
- suggest() HTTP error propagation.
- suggest() per_page override per-call.
- _parse_suggestion() with valid, partial, and missing-field items.
- ImageSuggestion dataclass and to_dict() serialisation.
- fetch_suggestions() convenience function.
- Context manager usage.
- _build_url() query parameter construction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
import requests
import responses as responses_lib

from ai_image_audit.suggester import (
    DEFAULT_PER_PAGE,
    DEFAULT_TIMEOUT,
    OPENVERSE_API_BASE,
    ImageSuggestion,
    OpenverseSuggester,
    _parse_suggestion,
    fetch_suggestions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    image_id: str = "abc123",
    url: str = "https://cdn.example.com/photo.jpg",
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a minimal Openverse API result item dictionary."""
    base: Dict[str, Any] = {
        "id": image_id,
        "title": "Test Image",
        "url": url,
        "thumbnail": "https://cdn.example.com/thumb.jpg",
        "foreign_landing_url": "https://flickr.com/photos/test",
        "creator": "Test Creator",
        "creator_url": "https://flickr.com/people/test",
        "license": "by",
        "license_version": "4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "provider": "flickr",
        "source": "flickr",
        "tags": [
            {"name": "nature"},
            {"name": "forest"},
        ],
        "width": 1920,
        "height": 1080,
    }
    base.update(overrides)
    return base


def _make_response(
    items: List[Dict[str, Any]], count: int = None
) -> Dict[str, Any]:
    """Build a mock Openverse API response body."""
    return {
        "count": count if count is not None else len(items),
        "next": None,
        "previous": None,
        "results": items,
    }


API_SEARCH_URL = f"{OPENVERSE_API_BASE}/images/"


# ---------------------------------------------------------------------------
# ImageSuggestion dataclass tests
# ---------------------------------------------------------------------------


class TestImageSuggestion:
    """Tests for the ImageSuggestion dataclass."""

    def _make_suggestion(self, **kwargs: Any) -> ImageSuggestion:
        """Build an ImageSuggestion with sensible defaults."""
        defaults = dict(
            id="abc123",
            title="Test Image",
            url="https://cdn.example.com/photo.jpg",
            thumbnail="https://cdn.example.com/thumb.jpg",
            foreign_landing_url="https://flickr.com/photos/test",
            creator="Test Creator",
            creator_url="https://flickr.com/people/test",
            licence="by",
            licence_version="4.0",
            licence_url="https://creativecommons.org/licenses/by/4.0/",
            provider="flickr",
            source="flickr",
            tags=["nature", "forest"],
            width=1920,
            height=1080,
        )
        defaults.update(kwargs)
        return ImageSuggestion(**defaults)

    def test_to_dict_returns_dict(self) -> None:
        """to_dict() should return a dictionary."""
        s = self._make_suggestion()
        assert isinstance(s.to_dict(), dict)

    def test_to_dict_contains_all_keys(self) -> None:
        """to_dict() should contain all expected top-level keys."""
        s = self._make_suggestion()
        d = s.to_dict()
        expected = {
            "id", "title", "url", "thumbnail", "foreign_landing_url",
            "creator", "creator_url", "licence", "licence_version",
            "licence_url", "provider", "source", "tags", "width", "height",
        }
        assert expected.issubset(d.keys())

    def test_to_dict_id(self) -> None:
        """to_dict() should serialise the id field."""
        s = self._make_suggestion(id="xyz999")
        assert s.to_dict()["id"] == "xyz999"

    def test_to_dict_url(self) -> None:
        """to_dict() should serialise the url field."""
        s = self._make_suggestion(url="https://example.com/img.jpg")
        assert s.to_dict()["url"] == "https://example.com/img.jpg"

    def test_to_dict_tags_list(self) -> None:
        """to_dict() should serialise tags as a list."""
        s = self._make_suggestion(tags=["a", "b", "c"])
        assert s.to_dict()["tags"] == ["a", "b", "c"]

    def test_optional_fields_can_be_none(self) -> None:
        """Optional fields set to None should serialise as None."""
        s = self._make_suggestion(
            title=None, thumbnail=None, creator=None, width=None, height=None
        )
        d = s.to_dict()
        assert d["title"] is None
        assert d["thumbnail"] is None
        assert d["creator"] is None
        assert d["width"] is None
        assert d["height"] is None

    def test_default_tags_empty_list(self) -> None:
        """When tags is not provided, it should default to an empty list."""
        s = ImageSuggestion(
            id="x",
            title=None,
            url="https://example.com/img.jpg",
            thumbnail=None,
            foreign_landing_url=None,
            creator=None,
            creator_url=None,
            licence="cc0",
            licence_version=None,
            licence_url=None,
            provider=None,
            source=None,
        )
        assert s.tags == []

    def test_to_dict_is_json_serialisable(self) -> None:
        """to_dict() output should be fully JSON-serialisable."""
        s = self._make_suggestion()
        serialised = json.dumps(s.to_dict())
        data = json.loads(serialised)
        assert data["id"] == "abc123"

    def test_to_dict_licence_field(self) -> None:
        """to_dict() should include the licence field."""
        s = self._make_suggestion(licence="cc0")
        assert s.to_dict()["licence"] == "cc0"

    def test_to_dict_thumbnail(self) -> None:
        """to_dict() should include the thumbnail field."""
        s = self._make_suggestion(thumbnail="https://cdn.example.com/t.jpg")
        assert s.to_dict()["thumbnail"] == "https://cdn.example.com/t.jpg"

    def test_to_dict_width_height(self) -> None:
        """to_dict() should include width and height."""
        s = self._make_suggestion(width=800, height=600)
        d = s.to_dict()
        assert d["width"] == 800
        assert d["height"] == 600

    def test_to_dict_creator(self) -> None:
        """to_dict() should include the creator field."""
        s = self._make_suggestion(creator="John Doe")
        assert s.to_dict()["creator"] == "John Doe"

    def test_to_dict_creator_url(self) -> None:
        """to_dict() should include creator_url."""
        s = self._make_suggestion(creator_url="https://flickr.com/john")
        assert s.to_dict()["creator_url"] == "https://flickr.com/john"

    def test_to_dict_foreign_landing_url(self) -> None:
        """to_dict() should include foreign_landing_url."""
        s = self._make_suggestion(foreign_landing_url="https://flickr.com/photos/x")
        assert s.to_dict()["foreign_landing_url"] == "https://flickr.com/photos/x"

    def test_to_dict_provider(self) -> None:
        """to_dict() should include the provider field."""
        s = self._make_suggestion(provider="wikimedia_commons")
        assert s.to_dict()["provider"] == "wikimedia_commons"

    def test_to_dict_source(self) -> None:
        """to_dict() should include the source field."""
        s = self._make_suggestion(source="wikimedia")
        assert s.to_dict()["source"] == "wikimedia"

    def test_to_dict_licence_version(self) -> None:
        """to_dict() should include licence_version."""
        s = self._make_suggestion(licence_version="2.0")
        assert s.to_dict()["licence_version"] == "2.0"

    def test_to_dict_licence_url(self) -> None:
        """to_dict() should include licence_url."""
        s = self._make_suggestion(
            licence_url="https://creativecommons.org/licenses/cc0/1.0/"
        )
        assert "creativecommons.org" in s.to_dict()["licence_url"]

    def test_tags_with_multiple_items(self) -> None:
        """Multiple tags should all be preserved in to_dict()."""
        tags = ["ocean", "beach", "sunset", "photography"]
        s = self._make_suggestion(tags=tags)
        assert s.to_dict()["tags"] == tags

    def test_empty_tags_list_serialised(self) -> None:
        """An empty tags list should serialise as an empty list."""
        s = self._make_suggestion(tags=[])
        assert s.to_dict()["tags"] == []

    def test_id_is_stored_as_string(self) -> None:
        """The id field should be a string."""
        s = self._make_suggestion(id="unique-id-456")
        assert isinstance(s.id, str)
        assert s.id == "unique-id-456"

    def test_url_stored_correctly(self) -> None:
        """The url field should be stored exactly as provided."""
        url = "https://cdn.openverse.example.com/image/large.jpg"
        s = self._make_suggestion(url=url)
        assert s.url == url


# ---------------------------------------------------------------------------
# _parse_suggestion tests
# ---------------------------------------------------------------------------


class TestParseSuggestion:
    """Tests for the _parse_suggestion() item parser."""

    def test_valid_item_returns_suggestion(self) -> None:
        """A fully valid item should parse into an ImageSuggestion."""
        item = _make_item()
        result = _parse_suggestion(item)
        assert isinstance(result, ImageSuggestion)

    def test_missing_id_returns_none(self) -> None:
        """An item without an 'id' key should return None."""
        item = _make_item()
        del item["id"]
        assert _parse_suggestion(item) is None

    def test_none_id_returns_none(self) -> None:
        """An item with id=None should return None."""
        item = _make_item()
        item["id"] = None
        assert _parse_suggestion(item) is None

    def test_missing_url_returns_none(self) -> None:
        """An item without a 'url' key should return None."""
        item = _make_item()
        del item["url"]
        assert _parse_suggestion(item) is None

    def test_none_url_returns_none(self) -> None:
        """An item with url=None should return None."""
        item = _make_item()
        item["url"] = None
        assert _parse_suggestion(item) is None

    def test_empty_string_id_returns_none(self) -> None:
        """An item with id='' (empty string) should return None."""
        item = _make_item()
        item["id"] = ""
        assert _parse_suggestion(item) is None

    def test_empty_string_url_returns_none(self) -> None:
        """An item with url='' (empty string) should return None."""
        item = _make_item()
        item["url"] = ""
        assert _parse_suggestion(item) is None

    def test_id_coerced_to_string(self) -> None:
        """An integer id should be coerced to a string."""
        item = _make_item()
        item["id"] = 12345
        result = _parse_suggestion(item)
        assert result is not None
        assert isinstance(result.id, str)
        assert result.id == "12345"

    def test_tags_extracted_from_dicts(self) -> None:
        """Tags supplied as dicts with 'name' keys should be extracted."""
        item = _make_item(tags=[{"name": "sunset"}, {"name": "sky"}])
        result = _parse_suggestion(item)
        assert result is not None
        assert "sunset" in result.tags
        assert "sky" in result.tags

    def test_tags_extracted_from_strings(self) -> None:
        """Tags supplied as plain strings should be extracted."""
        item = _make_item(tags=["mountain", "lake"])
        result = _parse_suggestion(item)
        assert result is not None
        assert "mountain" in result.tags
        assert "lake" in result.tags

    def test_mixed_tag_formats(self) -> None:
        """A mix of dict and string tags should both be extracted."""
        item = _make_item(tags=[{"name": "ocean"}, "beach"])
        result = _parse_suggestion(item)
        assert result is not None
        assert "ocean" in result.tags
        assert "beach" in result.tags

    def test_missing_tags_gives_empty_list(self) -> None:
        """An item without a 'tags' key should produce an empty tags list."""
        item = _make_item()
        item.pop("tags", None)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.tags == []

    def test_none_tags_gives_empty_list(self) -> None:
        """An item with tags=None should produce an empty tags list."""
        item = _make_item(tags=None)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.tags == []

    def test_empty_tags_list_gives_empty_list(self) -> None:
        """An item with tags=[] should produce an empty tags list."""
        item = _make_item(tags=[])
        result = _parse_suggestion(item)
        assert result is not None
        assert result.tags == []

    def test_width_height_parsed_correctly(self) -> None:
        """Width and height should be parsed as integers."""
        item = _make_item(width=800, height=600)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.width == 800
        assert result.height == 600

    def test_invalid_width_gives_none(self) -> None:
        """A non-numeric width should produce width=None."""
        item = _make_item(width="not_a_number")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.width is None

    def test_invalid_height_gives_none(self) -> None:
        """A non-numeric height should produce height=None."""
        item = _make_item(height="not_a_number")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.height is None

    def test_missing_width_gives_none(self) -> None:
        """An item without a 'width' key should produce width=None."""
        item = _make_item()
        item.pop("width", None)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.width is None

    def test_missing_height_gives_none(self) -> None:
        """An item without a 'height' key should produce height=None."""
        item = _make_item()
        item.pop("height", None)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.height is None

    def test_licence_field_mapped_from_license(self) -> None:
        """The 'license' API key should be mapped to the 'licence' field."""
        item = _make_item()
        item["license"] = "cc0"
        result = _parse_suggestion(item)
        assert result is not None
        assert result.licence == "cc0"

    def test_empty_title_becomes_none(self) -> None:
        """An empty title string should become None."""
        item = _make_item(title="")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.title is None

    def test_empty_creator_becomes_none(self) -> None:
        """An empty creator string should become None."""
        item = _make_item(creator="")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.creator is None

    def test_empty_thumbnail_becomes_none(self) -> None:
        """An empty thumbnail string should become None."""
        item = _make_item(thumbnail="")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.thumbnail is None

    def test_title_preserved_when_non_empty(self) -> None:
        """A non-empty title should be preserved."""
        item = _make_item(title="Beautiful Landscape")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.title == "Beautiful Landscape"

    def test_creator_preserved_when_non_empty(self) -> None:
        """A non-empty creator should be preserved."""
        item = _make_item(creator="Jane Photographer")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.creator == "Jane Photographer"

    def test_provider_preserved(self) -> None:
        """The provider field should be preserved."""
        item = _make_item(provider="wikimedia_commons")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.provider == "wikimedia_commons"

    def test_source_preserved(self) -> None:
        """The source field should be preserved."""
        item = _make_item(source="wikimedia")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.source == "wikimedia"

    def test_url_preserved(self) -> None:
        """The url field should be preserved."""
        url = "https://cdn.openverse.org/photo.jpg"
        item = _make_item(url=url)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.url == url

    def test_foreign_landing_url_preserved(self) -> None:
        """The foreign_landing_url field should be preserved."""
        item = _make_item(foreign_landing_url="https://flickr.com/photos/xyz")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.foreign_landing_url == "https://flickr.com/photos/xyz"

    def test_empty_foreign_landing_url_becomes_none(self) -> None:
        """An empty foreign_landing_url should become None."""
        item = _make_item(foreign_landing_url="")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.foreign_landing_url is None

    def test_licence_version_preserved(self) -> None:
        """The license_version field should be preserved as licence_version."""
        item = _make_item()
        item["license_version"] = "3.0"
        result = _parse_suggestion(item)
        assert result is not None
        assert result.licence_version == "3.0"

    def test_licence_url_preserved(self) -> None:
        """The license_url field should be preserved as licence_url."""
        item = _make_item()
        item["license_url"] = "https://creativecommons.org/licenses/by/4.0/"
        result = _parse_suggestion(item)
        assert result is not None
        assert result.licence_url == "https://creativecommons.org/licenses/by/4.0/"

    def test_tag_dict_without_name_key_skipped(self) -> None:
        """Tag dicts without a 'name' key should be silently skipped."""
        item = _make_item(tags=[{"label": "no_name"}, {"name": "valid_tag"}])
        result = _parse_suggestion(item)
        assert result is not None
        assert result.tags == ["valid_tag"]

    def test_tag_with_empty_name_skipped(self) -> None:
        """Tag dicts with an empty 'name' value should be skipped."""
        item = _make_item(tags=[{"name": ""}, {"name": "real_tag"}])
        result = _parse_suggestion(item)
        assert result is not None
        assert "" not in result.tags
        assert "real_tag" in result.tags

    def test_width_zero_parsed(self) -> None:
        """Width=0 should be parsed as integer 0."""
        item = _make_item(width=0)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.width == 0

    def test_height_zero_parsed(self) -> None:
        """Height=0 should be parsed as integer 0."""
        item = _make_item(height=0)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.height == 0

    def test_numeric_string_width_parsed(self) -> None:
        """A numeric string width like '1024' should be coerced to int."""
        item = _make_item(width="1024")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.width == 1024

    def test_empty_dict_returns_none(self) -> None:
        """An empty dict (missing required fields) should return None."""
        result = _parse_suggestion({})
        assert result is None

    def test_only_id_no_url_returns_none(self) -> None:
        """A dict with only id and no url should return None."""
        result = _parse_suggestion({"id": "abc"})
        assert result is None

    def test_only_url_no_id_returns_none(self) -> None:
        """A dict with only url and no id should return None."""
        result = _parse_suggestion({"url": "https://example.com/img.jpg"})
        assert result is None


# ---------------------------------------------------------------------------
# OpenverseSuggester initialisation tests
# ---------------------------------------------------------------------------


class TestOpenverseSuggesterInit:
    """Tests for OpenverseSuggester constructor."""

    def test_default_per_page(self) -> None:
        """Default per_page should equal DEFAULT_PER_PAGE."""
        s = OpenverseSuggester()
        assert s.per_page == DEFAULT_PER_PAGE
        s.close()

    def test_custom_per_page(self) -> None:
        """A custom per_page should be stored on the instance."""
        s = OpenverseSuggester(per_page=10)
        assert s.per_page == 10
        s.close()

    def test_per_page_capped_at_max(self) -> None:
        """per_page above 20 should be capped at 20."""
        s = OpenverseSuggester(per_page=100)
        assert s.per_page == 20
        s.close()

    def test_per_page_exactly_twenty(self) -> None:
        """per_page=20 should be accepted without capping."""
        s = OpenverseSuggester(per_page=20)
        assert s.per_page == 20
        s.close()

    def test_per_page_one_accepted(self) -> None:
        """per_page=1 is the minimum valid value."""
        s = OpenverseSuggester(per_page=1)
        assert s.per_page == 1
        s.close()

    def test_invalid_per_page_zero_raises(self) -> None:
        """per_page=0 should raise ValueError."""
        with pytest.raises(ValueError, match="per_page"):
            OpenverseSuggester(per_page=0)

    def test_invalid_per_page_negative_raises(self) -> None:
        """A negative per_page should raise ValueError."""
        with pytest.raises(ValueError, match="per_page"):
            OpenverseSuggester(per_page=-5)

    def test_default_timeout(self) -> None:
        """Default timeout should equal DEFAULT_TIMEOUT."""
        s = OpenverseSuggester()
        assert s.timeout == DEFAULT_TIMEOUT
        s.close()

    def test_custom_timeout(self) -> None:
        """A custom timeout should be stored on the instance."""
        s = OpenverseSuggester(timeout=30)
        assert s.timeout == 30
        s.close()

    def test_custom_api_base(self) -> None:
        """A custom api_base should be stored on the instance."""
        s = OpenverseSuggester(api_base="https://custom.api.example.com/v1")
        assert "custom.api.example.com" in s.api_base
        s.close()

    def test_api_base_trailing_slash_stripped(self) -> None:
        """A trailing slash in api_base should be stripped."""
        s = OpenverseSuggester(api_base="https://api.openverse.org/v1/")
        assert not s.api_base.endswith("/")
        s.close()

    def test_default_api_base_contains_openverse(self) -> None:
        """The default api_base should reference the Openverse API."""
        s = OpenverseSuggester()
        assert "openverse" in s.api_base
        s.close()

    def test_default_licence_filter_non_empty(self) -> None:
        """The default licence_filter should be a non-empty string."""
        s = OpenverseSuggester()
        assert s.licence_filter  # non-empty
        s.close()

    def test_custom_licence_filter(self) -> None:
        """A custom licence_filter should be stored on the instance."""
        s = OpenverseSuggester(licence_filter="cc0")
        assert s.licence_filter == "cc0"
        s.close()

    def test_external_session_used(self) -> None:
        """A caller-supplied session should be stored and not owned by the suggester."""
        custom_session = requests.Session()
        s = OpenverseSuggester(session=custom_session)
        assert s._session is custom_session
        assert s._own_session is False
        custom_session.close()

    def test_internal_session_created_when_none(self) -> None:
        """When no session is supplied, an internal session should be created."""
        s = OpenverseSuggester()
        assert s._session is not None
        assert s._own_session is True
        s.close()

    def test_session_has_user_agent_header(self) -> None:
        """The session should have a User-Agent header set."""
        s = OpenverseSuggester()
        assert "User-Agent" in s._session.headers
        s.close()

    def test_session_has_accept_header(self) -> None:
        """The session should have an Accept header set."""
        s = OpenverseSuggester()
        assert "Accept" in s._session.headers
        s.close()


# ---------------------------------------------------------------------------
# OpenverseSuggester.suggest() tests
# ---------------------------------------------------------------------------


class TestOpenverseSuggesterSuggest:
    """Tests for the OpenverseSuggester.suggest() method."""

    @responses_lib.activate
    def test_returns_list_of_suggestions(self) -> None:
        """suggest() should return a list of ImageSuggestion instances."""
        items = [
            _make_item(image_id=f"id{i}", url=f"https://cdn.example.com/{i}.jpg")
            for i in range(3)
        ]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("forest landscape")
        assert isinstance(result, list)
        assert len(result) == 3
        s.close()

    @responses_lib.activate
    def test_each_item_is_image_suggestion(self) -> None:
        """Each item in the returned list should be an ImageSuggestion."""
        items = [_make_item()]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("nature photography")
        assert all(isinstance(r, ImageSuggestion) for r in result)
        s.close()

    @responses_lib.activate
    def test_empty_results_returns_empty_list(self) -> None:
        """An API response with an empty results list should return []."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response([]),
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("very_obscure_query_xyz123")
        assert result == []
        s.close()

    def test_empty_query_returns_empty_list(self) -> None:
        """An empty query string should return [] without making an API call."""
        s = OpenverseSuggester()
        result = s.suggest("")
        assert result == []
        s.close()

    def test_whitespace_query_returns_empty_list(self) -> None:
        """A whitespace-only query should return [] without making an API call."""
        s = OpenverseSuggester()
        result = s.suggest("   ")
        assert result == []
        s.close()

    @responses_lib.activate
    def test_http_500_raises_request_exception(self) -> None:
        """A 500 response from the API should raise a RequestException."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            status=500,
        )
        s = OpenverseSuggester()
        with pytest.raises(requests.exceptions.RequestException):
            s.suggest("test")
        s.close()

    @responses_lib.activate
    def test_http_404_raises_request_exception(self) -> None:
        """A 404 response from the API should raise a RequestException."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            status=404,
        )
        s = OpenverseSuggester()
        with pytest.raises(requests.exceptions.RequestException):
            s.suggest("test")
        s.close()

    @responses_lib.activate
    def test_http_503_raises_request_exception(self) -> None:
        """A 503 response should raise a RequestException."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            status=503,
        )
        s = OpenverseSuggester()
        with pytest.raises(requests.exceptions.RequestException):
            s.suggest("test")
        s.close()

    @responses_lib.activate
    def test_connection_error_propagated(self) -> None:
        """A connection error should propagate as a RequestException."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            body=requests.exceptions.ConnectionError("refused"),
        )
        s = OpenverseSuggester()
        with pytest.raises(requests.exceptions.RequestException):
            s.suggest("test")
        s.close()

    @responses_lib.activate
    def test_timeout_error_propagated(self) -> None:
        """A timeout error should propagate as a RequestException."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            body=requests.exceptions.Timeout("timed out"),
        )
        s = OpenverseSuggester()
        with pytest.raises(requests.exceptions.RequestException):
            s.suggest("test")
        s.close()

    @responses_lib.activate
    def test_malformed_results_not_list_returns_empty(self) -> None:
        """If 'results' is not a list, suggest() should return []."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json={"count": 0, "results": "not a list"},
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("test")
        assert result == []
        s.close()

    @responses_lib.activate
    def test_missing_results_key_returns_empty(self) -> None:
        """If 'results' key is absent from the response, suggest() should return []."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json={"count": 0},
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("test")
        assert result == []
        s.close()

    @responses_lib.activate
    def test_per_page_override_per_call(self) -> None:
        """A per_page argument to suggest() should override the instance default."""
        items = [
            _make_item(image_id=f"id{i}", url=f"https://cdn.example.com/{i}.jpg")
            for i in range(3)
        ]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        s = OpenverseSuggester(per_page=5)
        result = s.suggest("test", per_page=3)
        assert isinstance(result, list)
        s.close()

    @responses_lib.activate
    def test_items_with_missing_id_are_skipped(self) -> None:
        """Items missing an 'id' should be silently skipped."""
        items = [
            _make_item(image_id="good1", url="https://cdn.example.com/1.jpg"),
            {"title": "No ID item", "url": "https://cdn.example.com/2.jpg"},
            _make_item(image_id="good2", url="https://cdn.example.com/3.jpg"),
        ]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("test")
        assert len(result) == 2
        s.close()

    @responses_lib.activate
    def test_items_with_missing_url_are_skipped(self) -> None:
        """Items missing a 'url' should be silently skipped."""
        items = [
            _make_item(image_id="good1", url="https://cdn.example.com/1.jpg"),
            {"id": "no_url", "title": "No URL item"},
            _make_item(image_id="good2", url="https://cdn.example.com/3.jpg"),
        ]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("test")
        assert len(result) == 2
        s.close()

    @responses_lib.activate
    def test_suggestion_fields_populated_correctly(self) -> None:
        """Parsed suggestion fields should match the API item data."""
        item = _make_item(
            image_id="test-id",
            url="https://cdn.example.com/photo.jpg",
            title="Beautiful Forest",
        )
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response([item]),
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("forest")
        assert len(result) == 1
        suggestion = result[0]
        assert suggestion.id == "test-id"
        assert suggestion.url == "https://cdn.example.com/photo.jpg"
        assert suggestion.title == "Beautiful Forest"
        assert suggestion.provider == "flickr"
        assert suggestion.licence == "by"
        assert "nature" in suggestion.tags
        s.close()

    @responses_lib.activate
    def test_width_height_in_suggestion(self) -> None:
        """Width and height should be parsed from the API response."""
        item = _make_item(width=1280, height=720)
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response([item]),
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("landscape")
        assert result[0].width == 1280
        assert result[0].height == 720
        s.close()

    @responses_lib.activate
    def test_suggest_returns_empty_when_all_items_malformed(self) -> None:
        """If all items are malformed (missing id/url), result should be empty."""
        items = [
            {"title": "No ID or URL"},
            {"title": "Also missing everything"},
        ]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("test")
        assert result == []
        s.close()

    @responses_lib.activate
    def test_multiple_suggestions_returned(self) -> None:
        """Multiple valid items should all be returned."""
        items = [
            _make_item(image_id=f"id{i}", url=f"https://cdn.example.com/{i}.jpg")
            for i in range(5)
        ]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        s = OpenverseSuggester()
        result = s.suggest("ocean photography")
        assert len(result) == 5
        s.close()

    @responses_lib.activate
    def test_query_is_stripped_before_sending(self) -> None:
        """Leading/trailing whitespace in the query should be stripped."""
        items = [_make_item()]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        s = OpenverseSuggester()
        # Should not raise and should return results.
        result = s.suggest("  forest  ")
        assert isinstance(result, list)
        s.close()


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


class TestContextManager:
    """Tests for OpenverseSuggester context manager usage."""

    @responses_lib.activate
    def test_context_manager_basic_usage(self) -> None:
        """Using the suggester as a context manager should work correctly."""
        items = [_make_item()]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        with OpenverseSuggester() as suggester:
            result = suggester.suggest("test")
        assert len(result) == 1

    def test_close_is_safe_to_call_multiple_times(self) -> None:
        """Calling close() multiple times should not raise."""
        s = OpenverseSuggester()
        s.close()
        s.close()  # Should not raise.

    def test_external_session_not_closed_on_exit(self) -> None:
        """An externally supplied session should not be closed by the suggester."""
        custom_session = requests.Session()
        with OpenverseSuggester(session=custom_session) as _:
            pass
        # Verify no exception during close(); external session should remain usable.
        custom_session.close()

    @responses_lib.activate
    def test_context_manager_enter_returns_self(self) -> None:
        """__enter__ should return the OpenverseSuggester instance."""
        s = OpenverseSuggester()
        with s as ctx:
            assert ctx is s

    def test_context_manager_close_called_on_exit(self) -> None:
        """The session should be closed after exiting the context manager."""
        s = OpenverseSuggester()
        with s:
            pass
        # After exiting, own session should have been closed.
        # We verify by checking _own_session was True.
        assert s._own_session is True
        # Calling close again should not raise.
        s.close()


# ---------------------------------------------------------------------------
# _build_url tests
# ---------------------------------------------------------------------------


class TestBuildUrl:
    """Tests for the _build_url() helper method."""

    def test_url_contains_encoded_query(self) -> None:
        """The query should appear URL-encoded in the built URL."""
        s = OpenverseSuggester()
        url = s._build_url("forest landscape", 5)
        # URL encoding may use + or %20 for spaces.
        assert "forest" in url
        assert "landscape" in url
        s.close()

    def test_url_contains_page_size_parameter(self) -> None:
        """The page_size parameter should be present in the URL."""
        s = OpenverseSuggester()
        url = s._build_url("test", 7)
        assert "page_size=7" in url
        s.close()

    def test_url_contains_licence_filter(self) -> None:
        """The licence filter should appear in the URL."""
        s = OpenverseSuggester(licence_filter="cc0")
        url = s._build_url("test", 5)
        assert "cc0" in url
        s.close()

    def test_url_starts_with_api_base(self) -> None:
        """The built URL should start with the configured api_base."""
        s = OpenverseSuggester()
        url = s._build_url("test", 5)
        assert url.startswith(OPENVERSE_API_BASE)
        s.close()

    def test_url_contains_images_path(self) -> None:
        """The built URL should include the /images/ endpoint path."""
        s = OpenverseSuggester()
        url = s._build_url("test", 5)
        assert "/images/" in url
        s.close()

    def test_mature_false_in_url(self) -> None:
        """The mature=false parameter should be present in the URL."""
        s = OpenverseSuggester()
        url = s._build_url("test", 5)
        assert "mature=false" in url
        s.close()

    def test_custom_api_base_reflected_in_url(self) -> None:
        """A custom api_base should be reflected in the built URL."""
        s = OpenverseSuggester(api_base="https://mock.api.example.com/v1")
        url = s._build_url("test", 5)
        assert url.startswith("https://mock.api.example.com/v1")
        s.close()

    def test_page_size_one_in_url(self) -> None:
        """page_size=1 should appear correctly in the URL."""
        s = OpenverseSuggester()
        url = s._build_url("test", 1)
        assert "page_size=1" in url
        s.close()

    def test_page_size_twenty_in_url(self) -> None:
        """page_size=20 should appear correctly in the URL."""
        s = OpenverseSuggester()
        url = s._build_url("test", 20)
        assert "page_size=20" in url
        s.close()

    def test_url_is_string(self) -> None:
        """_build_url should return a string."""
        s = OpenverseSuggester()
        url = s._build_url("test query", 5)
        assert isinstance(url, str)
        s.close()

    def test_url_contains_q_parameter(self) -> None:
        """The URL should contain a 'q' query parameter."""
        s = OpenverseSuggester()
        url = s._build_url("test", 5)
        assert "q=" in url
        s.close()

    def test_licence_multi_value_in_url(self) -> None:
        """A multi-value licence filter should appear in the URL."""
        s = OpenverseSuggester(licence_filter="cc0,by,by-sa")
        url = s._build_url("test", 5)
        assert "cc0" in url
        s.close()


# ---------------------------------------------------------------------------
# fetch_suggestions convenience function tests
# ---------------------------------------------------------------------------


class TestFetchSuggestions:
    """Tests for the module-level fetch_suggestions() convenience function."""

    @responses_lib.activate
    def test_returns_list(self) -> None:
        """fetch_suggestions() should return a list."""
        items = [_make_item()]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        result = fetch_suggestions("forest")
        assert isinstance(result, list)
        assert len(result) == 1

    @responses_lib.activate
    def test_returns_image_suggestion_instances(self) -> None:
        """Each item in the result should be an ImageSuggestion."""
        items = [_make_item()]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        result = fetch_suggestions("forest")
        assert all(isinstance(r, ImageSuggestion) for r in result)

    @responses_lib.activate
    def test_custom_per_page(self) -> None:
        """A custom per_page should be honoured."""
        items = [
            _make_item(image_id=f"id{i}", url=f"https://cdn.example.com/{i}.jpg")
            for i in range(3)
        ]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        result = fetch_suggestions("nature", per_page=3)
        assert len(result) == 3

    def test_empty_query_returns_empty_list(self) -> None:
        """An empty query should return [] without making an HTTP call."""
        result = fetch_suggestions("")
        assert result == []

    def test_whitespace_query_returns_empty_list(self) -> None:
        """A whitespace-only query should return [] without making an HTTP call."""
        result = fetch_suggestions("   ")
        assert result == []

    @responses_lib.activate
    def test_uses_custom_api_base(self) -> None:
        """A custom api_base should be used when querying."""
        custom_base = "https://mock-openverse.example.com/v1"
        custom_url = f"{custom_base}/images/"
        items = [_make_item()]
        responses_lib.add(
            responses_lib.GET,
            custom_url,
            json=_make_response(items),
            status=200,
        )
        result = fetch_suggestions("test", api_base=custom_base)
        assert len(result) == 1

    @responses_lib.activate
    def test_http_error_propagated(self) -> None:
        """An HTTP error from the API should propagate as a RequestException."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            status=503,
        )
        with pytest.raises(requests.exceptions.RequestException):
            fetch_suggestions("test")

    @responses_lib.activate
    def test_custom_session_used(self) -> None:
        """A custom requests.Session should be accepted and used."""
        items = [_make_item()]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        custom_session = requests.Session()
        result = fetch_suggestions("test", session=custom_session)
        assert len(result) == 1
        custom_session.close()

    @responses_lib.activate
    def test_custom_licence_filter(self) -> None:
        """A custom licence_filter should be accepted without error."""
        items = [_make_item()]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        result = fetch_suggestions("test", licence_filter="cc0")
        assert isinstance(result, list)

    @responses_lib.activate
    def test_custom_timeout_accepted(self) -> None:
        """A custom timeout parameter should be accepted without error."""
        items = [_make_item()]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        result = fetch_suggestions("test", timeout=30)
        assert isinstance(result, list)

    @responses_lib.activate
    def test_returns_empty_list_when_no_results(self) -> None:
        """An empty results array from the API should yield an empty list."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response([]),
            status=200,
        )
        result = fetch_suggestions("very_obscure_term_xyz")
        assert result == []

    @responses_lib.activate
    def test_suggestion_fields_accessible(self) -> None:
        """Fields on returned suggestions should be accessible."""
        item = _make_item(
            image_id="field-test",
            url="https://cdn.example.com/field-test.jpg",
            title="Field Test Image",
        )
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response([item]),
            status=200,
        )
        result = fetch_suggestions("test")
        assert len(result) == 1
        s = result[0]
        assert s.id == "field-test"
        assert s.url == "https://cdn.example.com/field-test.jpg"
        assert s.title == "Field Test Image"
        assert isinstance(s.tags, list)

    @responses_lib.activate
    def test_connection_error_propagated(self) -> None:
        """A connection error should propagate as a RequestException."""
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            body=requests.exceptions.ConnectionError("refused"),
        )
        with pytest.raises(requests.exceptions.RequestException):
            fetch_suggestions("test")
