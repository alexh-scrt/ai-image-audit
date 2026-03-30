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


def _make_response(items: List[Dict[str, Any]], count: int = None) -> Dict[str, Any]:
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
    def _make_suggestion(self, **kwargs: Any) -> ImageSuggestion:
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
        s = self._make_suggestion()
        assert isinstance(s.to_dict(), dict)

    def test_to_dict_contains_all_keys(self) -> None:
        s = self._make_suggestion()
        d = s.to_dict()
        expected = {
            "id", "title", "url", "thumbnail", "foreign_landing_url",
            "creator", "creator_url", "licence", "licence_version",
            "licence_url", "provider", "source", "tags", "width", "height",
        }
        assert expected.issubset(d.keys())

    def test_to_dict_id(self) -> None:
        s = self._make_suggestion(id="xyz999")
        assert s.to_dict()["id"] == "xyz999"

    def test_to_dict_url(self) -> None:
        s = self._make_suggestion(url="https://example.com/img.jpg")
        assert s.to_dict()["url"] == "https://example.com/img.jpg"

    def test_to_dict_tags_list(self) -> None:
        s = self._make_suggestion(tags=["a", "b", "c"])
        assert s.to_dict()["tags"] == ["a", "b", "c"]

    def test_optional_fields_can_be_none(self) -> None:
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
        s = self._make_suggestion()
        # Should not raise.
        serialised = json.dumps(s.to_dict())
        data = json.loads(serialised)
        assert data["id"] == "abc123"


# ---------------------------------------------------------------------------
# _parse_suggestion tests
# ---------------------------------------------------------------------------


class TestParseSuggestion:
    def test_valid_item_returns_suggestion(self) -> None:
        item = _make_item()
        result = _parse_suggestion(item)
        assert isinstance(result, ImageSuggestion)

    def test_missing_id_returns_none(self) -> None:
        item = _make_item()
        del item["id"]
        assert _parse_suggestion(item) is None

    def test_none_id_returns_none(self) -> None:
        item = _make_item(image_id=None)  # type: ignore[arg-type]
        assert _parse_suggestion(item) is None

    def test_missing_url_returns_none(self) -> None:
        item = _make_item()
        del item["url"]
        assert _parse_suggestion(item) is None

    def test_none_url_returns_none(self) -> None:
        item = _make_item(url=None)  # type: ignore[arg-type]
        assert _parse_suggestion(item) is None

    def test_id_coerced_to_string(self) -> None:
        item = _make_item(image_id=12345)
        result = _parse_suggestion(item)
        assert result is not None
        assert isinstance(result.id, str)
        assert result.id == "12345"

    def test_tags_extracted_from_dicts(self) -> None:
        item = _make_item(tags=[{"name": "sunset"}, {"name": "sky"}])
        result = _parse_suggestion(item)
        assert result is not None
        assert "sunset" in result.tags
        assert "sky" in result.tags

    def test_tags_extracted_from_strings(self) -> None:
        item = _make_item(tags=["mountain", "lake"])
        result = _parse_suggestion(item)
        assert result is not None
        assert "mountain" in result.tags
        assert "lake" in result.tags

    def test_missing_tags_gives_empty_list(self) -> None:
        item = _make_item()
        item.pop("tags", None)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.tags == []

    def test_none_tags_gives_empty_list(self) -> None:
        item = _make_item(tags=None)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.tags == []

    def test_width_height_parsed(self) -> None:
        item = _make_item(width=800, height=600)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.width == 800
        assert result.height == 600

    def test_invalid_width_gives_none(self) -> None:
        item = _make_item(width="not_a_number")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.width is None

    def test_missing_width_gives_none(self) -> None:
        item = _make_item()
        item.pop("width", None)
        result = _parse_suggestion(item)
        assert result is not None
        assert result.width is None

    def test_licence_field_mapped(self) -> None:
        item = _make_item(**{"license": "cc0"})
        result = _parse_suggestion(item)
        assert result is not None
        assert result.licence == "cc0"

    def test_empty_title_becomes_none(self) -> None:
        item = _make_item(title="")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.title is None

    def test_empty_creator_becomes_none(self) -> None:
        item = _make_item(creator="")
        result = _parse_suggestion(item)
        assert result is not None
        assert result.creator is None


# ---------------------------------------------------------------------------
# OpenverseSuggester initialisation tests
# ---------------------------------------------------------------------------


class TestOpenverseSuggesterInit:
    def test_default_per_page(self) -> None:
        s = OpenverseSuggester()
        assert s.per_page == DEFAULT_PER_PAGE

    def test_custom_per_page(self) -> None:
        s = OpenverseSuggester(per_page=10)
        assert s.per_page == 10

    def test_per_page_capped_at_max(self) -> None:
        s = OpenverseSuggester(per_page=100)
        assert s.per_page == 20

    def test_invalid_per_page_raises(self) -> None:
        with pytest.raises(ValueError, match="per_page"):
            OpenverseSuggester(per_page=0)

    def test_custom_timeout(self) -> None:
        s = OpenverseSuggester(timeout=30)
        assert s.timeout == 30

    def test_custom_api_base(self) -> None:
        s = OpenverseSuggester(api_base="https://custom.api.example.com/v1")
        assert "custom.api.example.com" in s.api_base

    def test_api_base_trailing_slash_stripped(self) -> None:
        s = OpenverseSuggester(api_base="https://api.openverse.org/v1/")
        assert not s.api_base.endswith("/")

    def test_default_licence_filter(self) -> None:
        s = OpenverseSuggester()
        assert s.licence_filter  # non-empty

    def test_custom_licence_filter(self) -> None:
        s = OpenverseSuggester(licence_filter="cc0")
        assert s.licence_filter == "cc0"

    def test_external_session_used(self) -> None:
        custom_session = requests.Session()
        s = OpenverseSuggester(session=custom_session)
        assert s._session is custom_session
        assert s._own_session is False
        custom_session.close()

    def test_internal_session_created(self) -> None:
        s = OpenverseSuggester()
        assert s._session is not None
        assert s._own_session is True
        s.close()


# ---------------------------------------------------------------------------
# OpenverseSuggester.suggest() tests
# ---------------------------------------------------------------------------


class TestOpenverseSuggesterSuggest:
    @responses_lib.activate
    def test_returns_list_of_suggestions(self) -> None:
        items = [_make_item(image_id=f"id{i}", url=f"https://cdn.example.com/{i}.jpg")
                 for i in range(3)]
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
        s = OpenverseSuggester()
        result = s.suggest("")
        assert result == []
        s.close()

    def test_whitespace_query_returns_empty_list(self) -> None:
        s = OpenverseSuggester()
        result = s.suggest("   ")
        assert result == []
        s.close()

    @responses_lib.activate
    def test_http_error_raises_request_exception(self) -> None:
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
    def test_404_raises_request_exception(self) -> None:
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
    def test_connection_error_propagated(self) -> None:
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
    def test_malformed_results_not_list_returns_empty(self) -> None:
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
    def test_per_page_override(self) -> None:
        items = [_make_item(image_id=f"id{i}", url=f"https://cdn.example.com/{i}.jpg")
                 for i in range(3)]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        s = OpenverseSuggester(per_page=5)
        # Override with a different per_page for this specific call.
        result = s.suggest("test", per_page=3)
        # The API mock returns 3 items; just check we called it.
        assert isinstance(result, list)
        s.close()

    @responses_lib.activate
    def test_items_with_missing_id_are_skipped(self) -> None:
        items = [
            _make_item(image_id="good1", url="https://cdn.example.com/1.jpg"),
            {"title": "No ID item", "url": "https://cdn.example.com/2.jpg"},  # missing id
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
        assert len(result) == 2  # The malformed item is skipped.
        s.close()

    @responses_lib.activate
    def test_suggestion_fields_populated(self) -> None:
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


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


class TestContextManager:
    @responses_lib.activate
    def test_context_manager_basic_usage(self) -> None:
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
        s = OpenverseSuggester()
        s.close()
        s.close()  # Should not raise.

    def test_external_session_not_closed_on_exit(self) -> None:
        custom_session = requests.Session()
        with OpenverseSuggester(session=custom_session) as _:
            pass
        # The session should still be usable (not closed) since it was external.
        # We cannot easily assert it's open without making a request, so just
        # verify no exception was raised during close().
        custom_session.close()


# ---------------------------------------------------------------------------
# _build_url tests
# ---------------------------------------------------------------------------


class TestBuildUrl:
    def test_url_contains_query(self) -> None:
        s = OpenverseSuggester()
        url = s._build_url("forest landscape", 5)
        assert "forest+landscape" in url or "forest%20landscape" in url or "forest landscape" in url
        s.close()

    def test_url_contains_page_size(self) -> None:
        s = OpenverseSuggester()
        url = s._build_url("test", 7)
        assert "page_size=7" in url
        s.close()

    def test_url_contains_licence_filter(self) -> None:
        s = OpenverseSuggester(licence_filter="cc0")
        url = s._build_url("test", 5)
        assert "cc0" in url
        s.close()

    def test_url_starts_with_api_base(self) -> None:
        s = OpenverseSuggester()
        url = s._build_url("test", 5)
        assert url.startswith(OPENVERSE_API_BASE)
        s.close()

    def test_url_contains_images_path(self) -> None:
        s = OpenverseSuggester()
        url = s._build_url("test", 5)
        assert "/images/" in url
        s.close()

    def test_mature_false_in_url(self) -> None:
        s = OpenverseSuggester()
        url = s._build_url("test", 5)
        assert "mature=false" in url
        s.close()


# ---------------------------------------------------------------------------
# fetch_suggestions convenience function tests
# ---------------------------------------------------------------------------


class TestFetchSuggestions:
    @responses_lib.activate
    def test_returns_list(self) -> None:
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
    def test_custom_per_page(self) -> None:
        items = [_make_item(image_id=f"id{i}", url=f"https://cdn.example.com/{i}.jpg")
                 for i in range(3)]
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            json=_make_response(items),
            status=200,
        )
        result = fetch_suggestions("nature", per_page=3)
        assert len(result) == 3

    def test_empty_query_returns_empty(self) -> None:
        result = fetch_suggestions("")
        assert result == []

    @responses_lib.activate
    def test_uses_custom_api_base(self) -> None:
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
        responses_lib.add(
            responses_lib.GET,
            API_SEARCH_URL,
            status=503,
        )
        with pytest.raises(requests.exceptions.RequestException):
            fetch_suggestions("test")

    @responses_lib.activate
    def test_custom_session_used(self) -> None:
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
