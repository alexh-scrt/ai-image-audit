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
from pathlib import Path
from typing import List

import pytest
import responses as responses_lib
import requests

from ai_image_audit.scanner import (
    DEFAULT_MAX_IMAGES,
    DEFAULT_REQUEST_TIMEOUT,
    SUPPORTED_EXTENSIONS,
    ImageRef,
    _is_url,
    _parse_srcset,
    _derive_base_url,
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
    (sub / "logo.svg").write_text("<svg/>")
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
    """Tests for the ImageRef dataclass and its __post_init__ logic."""

    def test_local_ref_extension_derived(self) -> None:
        """Extension should be derived from the local source path."""
        ref = ImageRef(source="/tmp/photo.JPG", is_remote=False, origin="/tmp")
        assert ref.extension == ".jpg"

    def test_remote_ref_extension_derived(self) -> None:
        """Extension should be derived from the URL path component."""
        ref = ImageRef(
            source="https://example.com/img/cat.PNG",
            is_remote=True,
            origin="https://example.com",
        )
        assert ref.extension == ".png"

    def test_extension_provided_not_overridden(self) -> None:
        """An explicitly provided extension should not be overridden by __post_init__."""
        ref = ImageRef(
            source="/img/photo.jpg",
            is_remote=False,
            origin="/img",
            extension=".jpeg",
        )
        assert ref.extension == ".jpeg"

    def test_remote_flag_true(self) -> None:
        """is_remote should be True for remote refs."""
        ref = ImageRef(
            source="https://example.com/a.png",
            is_remote=True,
            origin="https://example.com",
        )
        assert ref.is_remote is True

    def test_remote_flag_false(self) -> None:
        """is_remote should be False for local refs."""
        ref = ImageRef(source="/tmp/a.jpg", is_remote=False, origin="/tmp")
        assert ref.is_remote is False

    def test_alt_text_defaults_to_none(self) -> None:
        """alt_text should default to None when not supplied."""
        ref = ImageRef(source="/img/x.gif", is_remote=False, origin="/img")
        assert ref.alt_text is None

    def test_alt_text_stored(self) -> None:
        """Provided alt_text should be stored on the instance."""
        ref = ImageRef(
            source="https://example.com/a.png",
            is_remote=True,
            origin="https://example.com",
            alt_text="A cat",
        )
        assert ref.alt_text == "A cat"

    def test_no_extension_url(self) -> None:
        """URLs without a file extension should yield an empty extension string."""
        ref = ImageRef(
            source="https://example.com/api/img",
            is_remote=True,
            origin="https://example.com",
        )
        assert ref.extension == ""

    def test_extension_lowercase_for_local(self) -> None:
        """Derived extension should always be lowercase for local paths."""
        ref = ImageRef(source="/images/PHOTO.JPEG", is_remote=False, origin="/images")
        assert ref.extension == ".jpeg"

    def test_extension_lowercase_for_remote(self) -> None:
        """Derived extension should always be lowercase for remote URLs."""
        ref = ImageRef(
            source="https://cdn.example.com/BANNER.GIF",
            is_remote=True,
            origin="https://cdn.example.com",
        )
        assert ref.extension == ".gif"

    def test_origin_stored(self) -> None:
        """origin field should be stored on the instance."""
        ref = ImageRef(
            source="/tmp/images/photo.jpg",
            is_remote=False,
            origin="/tmp/images",
        )
        assert ref.origin == "/tmp/images"

    def test_source_stored(self) -> None:
        """source field should be stored on the instance."""
        ref = ImageRef(
            source="https://example.com/img.png",
            is_remote=True,
            origin="https://example.com",
        )
        assert ref.source == "https://example.com/img.png"

    def test_webp_extension_derived(self) -> None:
        """WebP extension should be derived correctly."""
        ref = ImageRef(source="/img/photo.webp", is_remote=False, origin="/img")
        assert ref.extension == ".webp"

    def test_tiff_extension_derived(self) -> None:
        """TIFF extension should be derived correctly."""
        ref = ImageRef(source="/img/scan.tiff", is_remote=False, origin="/img")
        assert ref.extension == ".tiff"

    def test_svg_extension_derived(self) -> None:
        """SVG extension should be derived correctly."""
        ref = ImageRef(source="/img/icon.svg", is_remote=False, origin="/img")
        assert ref.extension == ".svg"

    def test_url_with_query_string_extension_derived(self) -> None:
        """Extension should be derived from the URL path, ignoring query params."""
        ref = ImageRef(
            source="https://example.com/img/photo.jpg?size=large",
            is_remote=True,
            origin="https://example.com",
        )
        assert ref.extension == ".jpg"


# ---------------------------------------------------------------------------
# _is_url helper tests
# ---------------------------------------------------------------------------


class TestIsUrl:
    """Tests for the _is_url helper function."""

    @pytest.mark.parametrize(
        "value",
        [
            "http://example.com",
            "https://example.com/path",
            "HTTP://EXAMPLE.COM",
            "HTTPS://EXAMPLE.COM",
            "http://localhost:8080/page",
            "https://sub.domain.example.org/images/cat.jpg",
        ],
    )
    def test_valid_urls(self, value: str) -> None:
        """HTTP and HTTPS URLs should be recognised as URLs."""
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
            "file:///home/user/img.jpg",
            "data:image/png;base64,abc",
        ],
    )
    def test_non_urls(self, value: str) -> None:
        """Non-HTTP/HTTPS strings should not be recognised as URLs."""
        assert _is_url(value) is False


# ---------------------------------------------------------------------------
# _parse_srcset helper tests
# ---------------------------------------------------------------------------


class TestParseSrcset:
    """Tests for the _parse_srcset srcset attribute parser."""

    def test_empty_string(self) -> None:
        """An empty srcset string should return an empty list."""
        assert _parse_srcset("") == []

    def test_single_url_no_descriptor(self) -> None:
        """A single URL without a descriptor should be returned."""
        assert _parse_srcset("image.jpg") == ["image.jpg"]

    def test_single_url_with_width(self) -> None:
        """A single URL with a width descriptor should return just the URL."""
        assert _parse_srcset("image-300.jpg 300w") == ["image-300.jpg"]

    def test_multiple_candidates(self) -> None:
        """Multiple width-descriptor candidates should all be extracted."""
        srcset = "image-300.jpg 300w, image-600.jpg 600w, image-900.jpg 900w"
        result = _parse_srcset(srcset)
        assert result == ["image-300.jpg", "image-600.jpg", "image-900.jpg"]

    def test_pixel_density_descriptors(self) -> None:
        """Pixel-density descriptors (1x, 2x) should be stripped, returning URLs only."""
        srcset = "logo.png 1x, logo@2x.png 2x"
        result = _parse_srcset(srcset)
        assert result == ["logo.png", "logo@2x.png"]

    def test_absolute_urls(self) -> None:
        """Absolute URLs in srcset should be preserved."""
        srcset = "https://cdn.example.com/sm.jpg 480w, https://cdn.example.com/lg.jpg 800w"
        result = _parse_srcset(srcset)
        assert "https://cdn.example.com/sm.jpg" in result
        assert "https://cdn.example.com/lg.jpg" in result

    def test_whitespace_only(self) -> None:
        """A whitespace-only string should return an empty list."""
        assert _parse_srcset("   ") == []

    def test_single_url_no_descriptor_stripped(self) -> None:
        """URLs with surrounding whitespace should be stripped."""
        result = _parse_srcset("  image.jpg  ")
        assert result == ["image.jpg"]

    def test_three_widths(self) -> None:
        """Three width-descriptor candidates should all be extracted."""
        srcset = "small.jpg 320w, medium.jpg 768w, large.jpg 1280w"
        result = _parse_srcset(srcset)
        assert len(result) == 3
        assert "small.jpg" in result
        assert "medium.jpg" in result
        assert "large.jpg" in result

    def test_url_only_no_whitespace(self) -> None:
        """A single URL with no spaces should still be returned."""
        result = _parse_srcset("image.webp")
        assert result == ["image.webp"]


# ---------------------------------------------------------------------------
# _derive_base_url helper tests
# ---------------------------------------------------------------------------


class TestDeriveBaseUrl:
    """Tests for the _derive_base_url redirect-following helper."""

    def test_returns_response_url_when_available(self) -> None:
        """Should return the final URL from the response object."""
        mock_response = type(
            "MockResponse", (), {"url": "https://redirected.example.com/page"}
        )()
        result = _derive_base_url("https://original.example.com", mock_response)
        assert result == "https://redirected.example.com/page"

    def test_falls_back_to_requested_url_when_response_url_is_empty(self) -> None:
        """Should fall back to the requested URL when response.url is falsy."""
        mock_response = type("MockResponse", (), {"url": ""} )()
        result = _derive_base_url("https://original.example.com", mock_response)
        assert result == "https://original.example.com"

    def test_falls_back_to_requested_url_when_response_url_is_none(self) -> None:
        """Should fall back to the requested URL when response.url is None."""
        mock_response = type("MockResponse", (), {"url": None})()
        result = _derive_base_url("https://original.example.com", mock_response)
        assert result == "https://original.example.com"


# ---------------------------------------------------------------------------
# scan_directory tests
# ---------------------------------------------------------------------------


class TestScanDirectory:
    """Tests for the scan_directory() function."""

    def test_returns_list(self, image_dir: Path) -> None:
        """scan_directory should return a list."""
        result = scan_directory(str(image_dir))
        assert isinstance(result, list)

    def test_finds_top_level_images(self, image_dir: Path) -> None:
        """Top-level image files should be discovered."""
        sources = {ref.source for ref in scan_directory(str(image_dir))}
        assert str(image_dir / "photo.jpg") in sources
        assert str(image_dir / "banner.png") in sources

    def test_finds_nested_images(self, image_dir: Path) -> None:
        """Nested image files should be discovered recursively."""
        sources = {ref.source for ref in scan_directory(str(image_dir))}
        assert str(image_dir / "assets" / "icons" / "icon.webp") in sources
        assert str(image_dir / "assets" / "icons" / "logo.svg") in sources

    def test_excludes_non_image_files(self, image_dir: Path) -> None:
        """Non-image files should be excluded from results."""
        sources = {ref.source for ref in scan_directory(str(image_dir))}
        assert not any(s.endswith(".txt") for s in sources)
        assert not any(s.endswith(".py") for s in sources)
        assert not any(s.endswith(".json") for s in sources)

    def test_skips_node_modules(self, image_dir: Path) -> None:
        """Files inside node_modules should be skipped."""
        sources = {ref.source for ref in scan_directory(str(image_dir))}
        assert not any("node_modules" in s for s in sources)

    def test_skips_hidden_directories(self, image_dir: Path) -> None:
        """Files inside hidden directories (starting with '.') should be skipped."""
        sources = {ref.source for ref in scan_directory(str(image_dir))}
        assert not any(".hidden" in s for s in sources)

    def test_all_refs_are_local(self, image_dir: Path) -> None:
        """All discovered refs should have is_remote=False."""
        for ref in scan_directory(str(image_dir)):
            assert ref.is_remote is False

    def test_origin_is_root(self, image_dir: Path) -> None:
        """The origin field should be the resolved root directory."""
        for ref in scan_directory(str(image_dir)):
            assert ref.origin == str(image_dir.resolve())

    def test_extensions_are_lowercase(self, image_dir: Path) -> None:
        """All extension fields should be lowercase."""
        for ref in scan_directory(str(image_dir)):
            assert ref.extension == ref.extension.lower()

    def test_max_images_limit(self, image_dir: Path) -> None:
        """Results should not exceed the max_images limit."""
        result = scan_directory(str(image_dir), max_images=2)
        assert len(result) <= 2

    def test_missing_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        """A non-existent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            scan_directory(str(tmp_path / "nonexistent"))

    def test_file_path_raises_not_a_directory(self, tmp_path: Path) -> None:
        """Passing a file path instead of a directory should raise NotADirectoryError."""
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(NotADirectoryError):
            scan_directory(str(f))

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """An empty directory should return an empty list."""
        empty = tmp_path / "empty"
        empty.mkdir()
        result = scan_directory(str(empty))
        assert result == []

    def test_supported_extensions_covered(self, tmp_path: Path) -> None:
        """Every extension in SUPPORTED_EXTENSIONS should be discoverable."""
        for ext in SUPPORTED_EXTENSIONS:
            if ext == ".svg":
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

    def test_max_images_one_returns_single_result(self, image_dir: Path) -> None:
        """max_images=1 should return at most one result."""
        result = scan_directory(str(image_dir), max_images=1)
        assert len(result) == 1

    def test_all_sources_are_absolute_paths(self, image_dir: Path) -> None:
        """All source paths should be absolute filesystem paths."""
        for ref in scan_directory(str(image_dir)):
            assert os.path.isabs(ref.source)

    def test_alt_text_is_none_for_local_files(self, image_dir: Path) -> None:
        """alt_text should always be None for locally discovered images."""
        for ref in scan_directory(str(image_dir)):
            assert ref.alt_text is None

    def test_skips_git_directory(self, tmp_path: Path) -> None:
        """Files inside .git should be skipped."""
        git_dir = tmp_path / ".git" / "objects"
        git_dir.mkdir(parents=True)
        (git_dir / "cached.png").write_bytes(b"data")
        (tmp_path / "real.png").write_bytes(b"data")
        results = scan_directory(str(tmp_path))
        sources = [ref.source for ref in results]
        assert not any(".git" in s for s in sources)
        assert any("real.png" in s for s in sources)

    def test_skips_pycache_directory(self, tmp_path: Path) -> None:
        """Files inside __pycache__ should be skipped."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.png").write_bytes(b"data")
        (tmp_path / "real.jpg").write_bytes(b"data")
        results = scan_directory(str(tmp_path))
        sources = [ref.source for ref in results]
        assert not any("__pycache__" in s for s in sources)

    def test_skips_venv_directory(self, tmp_path: Path) -> None:
        """Files inside venv should be skipped."""
        venv = tmp_path / "venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "img.png").write_bytes(b"data")
        (tmp_path / "real.png").write_bytes(b"data")
        results = scan_directory(str(tmp_path))
        sources = [ref.source for ref in results]
        assert not any("venv" in s for s in sources)

    def test_relative_path_accepted(self, tmp_path: Path) -> None:
        """scan_directory should accept a relative path without raising."""
        (tmp_path / "a.jpg").write_bytes(b"data")
        # Change to parent so we can provide a relative path.
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path.parent))
            relative = tmp_path.name
            results = scan_directory(relative)
            assert len(results) == 1
        finally:
            os.chdir(original_cwd)

    def test_result_extension_matches_file(self, image_dir: Path) -> None:
        """Extension in ImageRef should match the actual file extension."""
        refs = scan_directory(str(image_dir))
        for ref in refs:
            actual_ext = Path(ref.source).suffix.lower()
            assert ref.extension == actual_ext

    def test_default_max_images_constant(self) -> None:
        """DEFAULT_MAX_IMAGES should be a positive integer."""
        assert isinstance(DEFAULT_MAX_IMAGES, int)
        assert DEFAULT_MAX_IMAGES > 0


# ---------------------------------------------------------------------------
# scan_url tests
# ---------------------------------------------------------------------------


class TestScanUrl:
    """Tests for the scan_url() function."""

    BASE_URL = "https://example.com"

    def _html(self, body: str) -> str:
        """Wrap a body string in a minimal HTML document."""
        return f"<html><head></head><body>{body}</body></html>"

    @responses_lib.activate
    def test_finds_img_src(self) -> None:
        """A simple <img src> should be discovered."""
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
        """URLs in an img srcset attribute should be discovered."""
        html = self._html(
            '<img src="/img/small.jpg" '
            'srcset="/img/small.jpg 400w, /img/large.jpg 800w" alt="">'
        )
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
        """URLs in <source srcset> inside <picture> should be discovered."""
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
        """An og:image meta tag should be discovered."""
        html = (
            "<html><head>"
            '<meta property="og:image" content="https://cdn.example.com/og.jpg">'
            "</head><body></body></html>"
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
        """The same URL appearing multiple times should only be returned once."""
        html = self._html(
            '<img src="/img/cat.jpg">'
            '<img src="/img/cat.jpg">'
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
        """Results should not exceed the max_images limit."""
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
        """Relative image URLs should be resolved against the page URL."""
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
    def test_filters_non_image_extensions(self) -> None:
        """Files with unsupported extensions (e.g. .pdf) should be excluded."""
        html = self._html(
            '<img src="/api/img/123">'
            '<img src="/file.pdf">'
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
        """A 4xx/5xx HTTP response should raise an HTTPError."""
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            status=404,
        )
        with pytest.raises(requests.exceptions.HTTPError):
            scan_url(self.BASE_URL)

    @responses_lib.activate
    def test_connection_error_raises(self) -> None:
        """A connection error should propagate as a ConnectionError."""
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=requests.exceptions.ConnectionError("Connection refused"),
        )
        with pytest.raises(requests.exceptions.ConnectionError):
            scan_url(self.BASE_URL)

    def test_non_url_raises_value_error(self) -> None:
        """Passing a non-URL string should raise ValueError."""
        with pytest.raises(ValueError):
            scan_url("/local/path")

    @responses_lib.activate
    def test_empty_page_returns_empty_list(self) -> None:
        """A page with no image tags should return an empty list."""
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
        """All discovered refs should have is_remote=True."""
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
        """The origin field should be set to the scanned URL."""
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

    @responses_lib.activate
    def test_500_error_raises_http_error(self) -> None:
        """A 500 HTTP response should raise an HTTPError."""
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            status=500,
        )
        with pytest.raises(requests.exceptions.HTTPError):
            scan_url(self.BASE_URL)

    @responses_lib.activate
    def test_absolute_img_src_preserved(self) -> None:
        """An absolute img src URL should be preserved as-is."""
        html = self._html('<img src="https://cdn.example.com/photo.jpg">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        assert result[0].source == "https://cdn.example.com/photo.jpg"

    @responses_lib.activate
    def test_no_extension_url_included(self) -> None:
        """Image URLs with no file extension should be included (dynamic endpoints)."""
        html = self._html('<img src="/api/image/123">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        assert len(result) == 1
        assert result[0].extension == ""

    @responses_lib.activate
    def test_multiple_images_discovered(self) -> None:
        """Multiple distinct image URLs should all be discovered."""
        html = self._html(
            '<img src="/a.jpg">'
            '<img src="/b.png">'
            '<img src="/c.webp">'
        )
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        assert len(result) == 3

    @responses_lib.activate
    def test_fragment_stripped_from_url(self) -> None:
        """Fragment identifiers should be stripped from image URLs."""
        html = self._html('<img src="/img/photo.jpg#section">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        assert len(result) == 1
        assert "#" not in result[0].source

    @responses_lib.activate
    def test_alt_text_empty_string_stored_as_none(self) -> None:
        """An empty alt attribute should result in alt_text=None."""
        html = self._html('<img src="/img/photo.jpg" alt="">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL)
        assert result[0].alt_text is None

    @responses_lib.activate
    def test_max_images_zero_returns_empty(self) -> None:
        """max_images=0 should return an empty list."""
        html = self._html('<img src="/img/photo.jpg">')
        responses_lib.add(
            responses_lib.GET,
            self.BASE_URL,
            body=html,
            content_type="text/html",
        )
        result = scan_url(self.BASE_URL, max_images=0)
        assert result == []


# ---------------------------------------------------------------------------
# Top-level scan() dispatcher tests
# ---------------------------------------------------------------------------


class TestScan:
    """Tests for the top-level scan() dispatcher function."""

    def test_dispatches_to_directory_for_local_path(self, tmp_path: Path) -> None:
        """A local path should be dispatched to scan_directory."""
        (tmp_path / "img.png").write_bytes(b"data")
        result = scan(str(tmp_path))
        assert len(result) == 1
        assert result[0].is_remote is False

    @responses_lib.activate
    def test_dispatches_to_url_for_http(self) -> None:
        """An http:// URL should be dispatched to scan_url."""
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

    @responses_lib.activate
    def test_dispatches_to_url_for_https(self) -> None:
        """An https:// URL should be dispatched to scan_url."""
        url = "https://secure.example.com"
        responses_lib.add(
            responses_lib.GET,
            url,
            body="<html><body><img src='/b.png'></body></html>",
            content_type="text/html",
        )
        result = scan(url)
        assert len(result) == 1
        assert result[0].is_remote is True

    def test_empty_target_raises_value_error(self) -> None:
        """An empty target string should raise ValueError."""
        with pytest.raises(ValueError):
            scan("")

    def test_whitespace_only_raises_value_error(self) -> None:
        """A whitespace-only target should raise ValueError."""
        with pytest.raises(ValueError):
            scan("   ")

    def test_missing_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        """A target pointing to a non-existent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            scan(str(tmp_path / "does_not_exist"))

    @responses_lib.activate
    def test_max_images_passed_through_local(self, tmp_path: Path) -> None:
        """max_images parameter should be respected for local scans."""
        for i in range(10):
            (tmp_path / f"img{i}.jpg").write_bytes(b"data")
        result = scan(str(tmp_path), max_images=3)
        assert len(result) <= 3

    @responses_lib.activate
    def test_max_images_passed_through_url(self) -> None:
        """max_images parameter should be respected for URL scans."""
        url = "https://example.com"
        imgs = "".join(f'<img src="/img/photo{i}.jpg">' for i in range(20))
        html = f"<html><body>{imgs}</body></html>"
        responses_lib.add(
            responses_lib.GET,
            url,
            body=html,
            content_type="text/html",
        )
        result = scan(url, max_images=5)
        assert len(result) <= 5

    def test_strips_whitespace_from_target(self, tmp_path: Path) -> None:
        """Leading/trailing whitespace in the target should be stripped."""
        (tmp_path / "photo.jpg").write_bytes(b"data")
        # Should not raise and should find the image.
        result = scan(f"  {tmp_path}  ")
        assert len(result) == 1

    def test_result_is_list(self, tmp_path: Path) -> None:
        """scan() should always return a list."""
        result = scan(str(tmp_path))
        assert isinstance(result, list)

    def test_file_as_target_raises_not_a_directory(self, tmp_path: Path) -> None:
        """Passing a file path (not a directory) should raise NotADirectoryError."""
        f = tmp_path / "image.jpg"
        f.write_bytes(b"data")
        with pytest.raises(NotADirectoryError):
            scan(str(f))

    @responses_lib.activate
    def test_request_timeout_passed_through(self) -> None:
        """The request_timeout parameter should be accepted without error."""
        url = "https://example.com"
        responses_lib.add(
            responses_lib.GET,
            url,
            body="<html><body></body></html>",
            content_type="text/html",
        )
        result = scan(url, request_timeout=5)
        assert isinstance(result, list)

    def test_default_request_timeout_constant(self) -> None:
        """DEFAULT_REQUEST_TIMEOUT should be a positive integer."""
        assert isinstance(DEFAULT_REQUEST_TIMEOUT, int)
        assert DEFAULT_REQUEST_TIMEOUT > 0
