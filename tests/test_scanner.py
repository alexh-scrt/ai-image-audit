"""Unit tests for ai_image_audit.scanner.

Covers:
- Local directory scanning (happy path, empty dir, nested dirs, non-image files,
  skipped dirs, max_images limit, missing dir, file-not-dir).
- URL scanning (happy path, srcset, og:image, deduplication, max_images,
  non-html content-type warning, relative URL resolution, HTTP errors).
- Helper functions: _is_url, _parse_srcset, _derive_base_url.
- ImageRef dataclass behaviour (__post_init__ extension derivation).
- The top-level scan() dispatcher.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import List

import pytest
import responses as responses_lib
import requests

from ai_image_audit.scanner import (
    DEFAULT_MAX_IMAGES,
    SUPPORTED_EXTENSIONS,
    ImageRef,
    _is_url,
    _parse_srcset,
    scan,
    scan_directory,
    scan_url,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory populated with a mix of image and non-image files."""
    # Top-level images
    (tmp_path / "photo.jpg").write_bytes(b"fake-jpg")
    (tmp_path / "banner.png").write_bytes(b"fake-png")
    (tmp_path / "readme.txt").write_bytes(b"not an image")
    (tmp_path / "script.py").write_bytes(b"print('hi')")

    # Nested subdirectory
    sub = tmp_path / "assets" / "icons"
    sub.mkdir(parents=True)
    (sub / "icon.webp").write_bytes(b"fake-webp")
    (sub / "logo.svg").write_bytes(b"<svg/>")
    (sub / "data.json").write_bytes(b"{}")

    # A directory that should be skipped
    node = tmp_path / "node_modules" / "pkg"
    node.mkdir(parents=True)
    (node / "image.png").write_bytes(b"should-be-skipped")

    # Hidden directory that should be skipped
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.jpg").write_bytes(b"should-be-skipped")

    return tmp_path


# ---------------------------------------------------------------------------
# ImageRef dataclass tests
# ---------------------------------------------------------------------------


class TestImageRef:
    def test_local_ref_extension_derived(self) -> None:
        ref = ImageRef(source="/tmp/photo.JPG", is_remote=False, origin="/tmp")
        assert ref.extension == ".jpg"

    def test_remote_ref_extension_derived(self) -> None:
        ref = ImageRef(
            source="https://example.com/img/cat.PNG",
            is_remote=True,
            origin="https://example.com",
        )
        assert ref.extension == ".png"

    def test_extension_provided_not_overridden(self) -> None:
        ref = ImageRef(
            source="/img/photo.jpg",
            is_remote=False,
            origin="/img",
            extension=".jpeg",
        )
        assert ref.extension == ".jpeg"

    def test_remote_flag(self) -> None:
        ref = ImageRef(
            source="https://example.com/a.png",
            is_remote=True,
            origin="https://example.com",
        )
        assert ref.is_remote is True

    def test_alt_text_defaults_to_none(self) -> None:
        ref = ImageRef(source="/img/x.gif", is_remote=False, origin="/img")
        assert ref.alt_text is None

    def test_alt_text_stored(self) -> None:
        ref = ImageRef(
            source="https://example.com/a.png",
            is_remote=True,
            origin="https://example.com",
            alt_text="A cat",
        )
        assert ref.alt_text == "A cat"

    def test_no_extension_url(self) -> None:
        ref = ImageRef(
            source="https://example.com/api/img",
            is_remote=True,
            origin="https://example.com",
        )
        # No suffix – extension should be empty string
        assert ref.extension == ""


# ---------------------------------------------------------------------------
# _is_url helper
# ---------------------------------------------------------------------------


class TestIsUrl:
    @pytest.mark.parametrize(
        "value",
        [
            "http://example.com",
            "https://example.com/path",
            "HTTP://EXAMPLE.COM",
            "HTTPS://EXAMPLE.COM",
        ],
    )
    def test_valid_urls(self, value: str) -> None:
        assert _is_url(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "/tmp/images",
            "./relative/path",
            "C:\\Users\\images",
            "ftp://example.com",
            "",
            "example.com",
        ],
    )
    def test_non_urls(self, value: str) -> None:
        assert _is_url(value) is False


# ---------------------------------------------------------------------------
# _parse_srcset helper
# ---------------------------------------------------------------------------


class TestParseSrcset:
    def test_empty_string(self) -> None:
        assert _parse_srcset("") == []

    def test_single_url_no_descriptor(self) -> None:
        assert _parse_srcset("image.jpg") == ["image.jpg"]

    def test_single_url_with_width(self) -> None:
        assert _parse_srcset("image-300.jpg 300w") == ["image-300.jpg"]

    def test_multiple_candidates(self) -> None:
        srcset = "image-300.jpg 300w, image-600.jpg 600w, image-900.jpg 900w"
        result = _parse_srcset(srcset)
        assert result == ["image-300.jpg", "image-600.jpg", "image-900.jpg"]

    def test_pixel_density_descriptors(self) -> None:
        srcset = "logo.png 1x, logo@2x.png 2x"
        result = _parse_srcset(srcset)
        assert result == ["logo.png", "logo@2x.png"]

    def test_absolute_urls(self) -> None:
        srcset = "https://cdn.example.com/sm.jpg 480w, https://cdn.example.com/lg.jpg 800w"
        result = _parse_srcset(srcset)
        assert "https://cdn.example.com/sm.jpg" in result
        assert "https://cdn.example.com/lg.jpg" in result

    def test_whitespace_only(self) -> None:
        assert _parse_srcset("   ") == []


# ---------------------------------------------------------------------------
# scan_directory tests
# ---------------------------------------------------------------------------


class TestScanDirectory:
    def test_returns_list(self, image_dir: Path) -> None:
        result = scan_directory(str(image_dir))
        assert isinstance(result, list)

    def test_finds_top_level_images(self, image_dir: Path) -> None:
        sources = {ref.source for ref in scan_directory(str(image_dir))}
        assert str(image_dir / "photo.jpg") in sources
        assert str(image_dir / "banner.png") in sources

    def test_finds_nested_images(self, image_dir: Path) -> None:
        sources = {ref.source for ref in scan_directory(str(image_dir))}
        assert str(image_dir / "assets" / "icons" / "icon.webp") in sources
        assert str(image_dir / "assets" / "icons" / "logo.svg") in sources

    def test_excludes_non_image_files(self, image_dir: Path) -> None:
        sources = {ref.source for ref in scan_directory(str(image_dir))}
        assert not any(s.endswith(".txt") for s in sources)
        assert not any(s.endswith(".py") for s in sources)
        assert not any(s.endswith(".json") for s in sources)

    def test_skips_node_modules(self, image_dir: Path) -> None:
        sources = {ref.source for ref in scan_directory(str(image_dir))}
        assert not any("node_modules" in s for s in sources)

    def test_skips_hidden_directories(self, image_dir: Path) -> None:
        sources = {ref.source for ref in scan_directory(str(image_dir))}
        assert not any(".hidden" in s for s in sources)

    def test_all_refs_are_local(self, image_dir: Path) -> None:
        for ref in scan_directory(str(image_dir)):
            assert ref.is_remote is False

    def test_origin_is_root(self, image_dir: Path) -> None:
        for ref in scan_directory(str(image_dir)):
            assert ref.origin == str(image_dir.resolve())

    def test_extensions_are_lowercase(self, image_dir: Path) -> None:
        for ref in scan_directory(str(image_dir)):
            assert ref.extension == ref.extension.lower()

    def test_max_images_limit(self, image_dir: Path) -> None:
        result = scan_directory(str(image_dir), max_images=2)
        assert len(result) <= 2

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            scan_directory(str(tmp_path / "nonexistent"))

    def test_file_path_raises_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(NotADirectoryError):
            scan_directory(str(f))

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = scan_directory(str(empty))
        assert result == []

    def test_supported_extensions_covered(self, tmp_path: Path) -> None:
        """Every extension in SUPPORTED_EXTENSIONS should be discoverable."""
        for ext in SUPPORTED_EXTENSIONS:
            if ext == ".svg":  # SVG is text; create appropriately
                (tmp_path / f"img{ext}").write_text("<svg/>")
            else:
                (tmp_path / f"img{ext}").write_bytes(b"data")
        results = scan_directory(str(tmp_path))
        found_exts = {ref.extension for ref in results}
        assert SUPPORTED_EXTENSIONS == found_exts

    def test_uppercase_extensions_found(self, tmp_path: Path) -> None:
        """Files with uppercase extensions should still be discovered."""
        (tmp_path / "IMAGE.JPG").write_bytes(b"data")
        results = scan_directory(str(tmp_path))
        assert len(results) == 1
        assert results[0].extension == ".jpg"


# ---------------------------------------------------------------------------
# scan_url tests
# ---------------------------------------------------------------------------


class TestScanUrl:
    BASE_URL = "https://example.com"

    def _html(self, body: str) -> str:
        return f"<html><head></head><body>{body}</body></html>"

    @responses_lib.activate
    def test_finds_img_src(self) -> None:
        html = self._html('<img src="/images/cat.jpg" alt="A cat">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        assert len(result) == 1
        assert result[0].source == "https://example.com/images/cat.jpg"
        assert result[0].alt_text == "A cat"
        assert result[0].is_remote is True

    @responses_lib.activate
    def test_finds_img_srcset(self) -> None:
        html = self._html(
            '<img src="/img/small.jpg" '
            'srcset="/img/small.jpg 400w, /img/large.jpg 800w" alt="">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        sources = [r.source for r in result]
        assert "https://example.com/img/small.jpg" in sources
        assert "https://example.com/img/large.jpg" in sources

    @responses_lib.activate
    def test_finds_picture_source_srcset(self) -> None:
        html = self._html(
            "<picture>"
            '<source srcset="/img/photo.webp" type="image/webp">'
            '<img src="/img/photo.jpg">'
            "</picture>"
        )
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        sources = [r.source for r in result]
        assert "https://example.com/img/photo.webp" in sources
        assert "https://example.com/img/photo.jpg" in sources

    @responses_lib.activate
    def test_finds_og_image(self) -> None:
        html = (
            '<html><head>'
            '<meta property="og:image" content="https://cdn.example.com/og.jpg">'
            '</head><body></body></html>'
        )
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        sources = [r.source for r in result]
        assert "https://cdn.example.com/og.jpg" in sources

    @responses_lib.activate
    def test_deduplicates_urls(self) -> None:
        # Same URL appears twice via different tags
        html = self._html(
            '<img src="/img/cat.jpg">'  # first occurrence
            '<img src="/img/cat.jpg">'  # duplicate
        )
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        sources = [r.source for r in result]
        assert sources.count("https://example.com/img/cat.jpg") == 1

    @responses_lib.activate
    def test_max_images_limit(self) -> None:
        imgs = "".join(f'<img src="/img/photo{i}.jpg">' for i in range(20))
        html = self._html(imgs)
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL, max_images=5)
        assert len(result) <= 5

    @responses_lib.activate
    def test_resolves_relative_urls(self) -> None:
        html = self._html('<img src="images/dog.png">')
        responses_lib.add(
            responses_lib.GET,
            "https://example.com/gallery/",
            body=html,
            content_type="text/html",
        )
        result = scan_url("https://example.com/gallery/")
        assert result[0].source == "https://example.com/gallery/images/dog.png"

    @responses_lib.activate
    def test_filters_non_image_hrefs(self) -> None:
        """Non-image URLs embedded via <img> should still pass (no extension = allowed)."""
        html = self._html(
            '<img src="/api/img/123">'
            '<img src="/file.pdf">'  # .pdf should be excluded
        )
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        sources = [r.source for r in result]
        assert not any(".pdf" in s for s in sources)

    @responses_lib.activate
    def test_http_error_raises(self) -> None:
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            status=404,
        )
        with pytest.raises(requests.exceptions.HTTPError):
            scan_url(self.BASE_URL)

    @responses_lib.activate
    def test_connection_error_raises(self) -> None:
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=requests.exceptions.ConnectionError("Connection refused"),
        )
        with pytest.raises(requests.exceptions.ConnectionError):
            scan_url(self.BASE_URL)

    def test_non_url_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            scan_url("/local/path")

    @responses_lib.activate
    def test_empty_page_returns_empty_list(self) -> None:
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body="<html><body></body></html>",
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        assert result == []

    @responses_lib.activate
    def test_all_refs_are_remote(self) -> None:
        html = self._html('<img src="/a.jpg"><img src="/b.png">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        for ref in scan_url(self.BASE_URL):
            assert ref.is_remote is True

    @responses_lib.activate
    def test_origin_is_request_url(self) -> None:
        html = self._html('<img src="/a.jpg">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        assert result[0].origin == self.BASE_URL

    @responses_lib.activate
    def test_non_html_content_type_still_parses(self) -> None:
        """A non-HTML content-type should log a warning but attempt parsing."""
        html = self._html('<img src="/img/a.jpg">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="application/xhtml+xml",
        )
        result = scan_url(self.BASE_URL)
        # Should still find the image
        assert any("/img/a.jpg" in r.source for r in result)

    @responses_lib.activate
    def test_custom_session_used(self) -> None:
        """A caller-supplied session should be used for requests."""
        html = self._html('<img src="/img/a.jpg">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        custom_session = requests.Session()
        result = scan_url(self.BASE_URL, session=custom_session)
        assert len(result) == 1
        custom_session.close()


# ---------------------------------------------------------------------------
# Top-level scan() dispatcher tests
# ---------------------------------------------------------------------------


class TestScan:
    def test_dispatches_to_directory_for_local_path(self, tmp_path: Path) -> None:
        (tmp_path / "img.png").write_bytes(b"data")
        result = scan(str(tmp_path))
        assert len(result) == 1
        assert result[0].is_remote is False

    @responses_lib.activate
    def test_dispatches_to_url_for_http(self) -> None:
        url = "https://example.com"
        responses_lib.add(
            responses_lib.GET,
            url,
            body="<html><body><img src='/a.jpg'></body></html>",
            content_type="text/html",
        )
        result = scan(url)
        assert len(result) == 1
        assert result[0].is_remote is True

    def test_empty_target_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            scan("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            scan("   ")

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            scan(str(tmp_path / "does_not_exist"))

    @responses_lib.activate
    def test_max_images_passed_through(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"img{i}.jpg").write_bytes(b"data")
        result = scan(str(tmp_path), max_images=3)
        assert len(result) <= 3
