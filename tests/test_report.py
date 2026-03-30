"""Unit tests for ai_image_audit.report.

Covers:
- ColourStats dataclass and to_dict() serialisation.
- ImageReport dataclass and to_dict() serialisation.
- compute_colour_stats() with various image modes, sizes, and edge cases.
- generate_report() with valid inputs, type errors, and edge-case images.
- generate_report_from_path() with valid, missing, and corrupt files.
- _get_file_size() for local and remote refs.
- _extract_dominant_colours() for small/minimal images.
- _to_rgb_safe() mode conversion helper.
- Error report generation via generate_report_from_path().
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from PIL import Image

from ai_image_audit.classifier import ClassificationResult
from ai_image_audit.report import (
    DEFAULT_PALETTE_COLOURS,
    ColourStats,
    ImageReport,
    _extract_dominant_colours,
    _get_file_size,
    _to_rgb_safe,
    compute_colour_stats,
    generate_report,
    generate_report_from_path,
)
from ai_image_audit.scanner import ImageRef


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_ref(tmp_path: Path) -> ImageRef:
    """A local ImageRef pointing to a real JPEG file."""
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 80), color=(200, 100, 50)).save(str(img_path), format="JPEG")
    return ImageRef(
        source=str(img_path),
        is_remote=False,
        origin=str(tmp_path),
        alt_text=None,
        extension=".jpg",
    )


@pytest.fixture()
def remote_ref() -> ImageRef:
    """A remote ImageRef (no local file exists)."""
    return ImageRef(
        source="https://example.com/images/sunset.jpg",
        is_remote=True,
        origin="https://example.com",
        alt_text="A beautiful sunset",
        extension=".jpg",
    )


@pytest.fixture()
def classification() -> ClassificationResult:
    """A synthetic ClassificationResult."""
    return ClassificationResult(
        score=0.72,
        is_flagged=True,
        threshold=0.5,
        verdict="AI-generated (flagged)",
        model_mode="heuristic",
    )


@pytest.fixture()
def rgb_image() -> Image.Image:
    """A small 100x80 solid-colour RGB image."""
    return Image.new("RGB", (100, 80), color=(200, 100, 50))


@pytest.fixture()
def rgba_image() -> Image.Image:
    """A small RGBA image."""
    return Image.new("RGBA", (64, 64), color=(0, 128, 255, 200))


@pytest.fixture()
def grayscale_image() -> Image.Image:
    """A small grayscale image."""
    return Image.new("L", (32, 32), color=128)


@pytest.fixture()
def tiny_image() -> Image.Image:
    """A 1x1 pixel image."""
    return Image.new("RGB", (1, 1), color=(255, 0, 0))


@pytest.fixture()
def wide_image() -> Image.Image:
    """A wide (panoramic) image to test aspect ratio."""
    return Image.new("RGB", (1920, 400), color=(100, 150, 200))


@pytest.fixture()
def valid_jpeg_path(tmp_path: Path) -> Path:
    """Save a valid JPEG to a temp dir and return its path."""
    p = tmp_path / "photo.jpg"
    Image.new("RGB", (64, 64), color=(50, 100, 150)).save(str(p), format="JPEG")
    return p


@pytest.fixture()
def corrupt_file_path(tmp_path: Path) -> Path:
    """Create a file with non-image content."""
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image at all")
    return p


@pytest.fixture()
def la_image() -> Image.Image:
    """A small LA (grayscale + alpha) image."""
    return Image.new("LA", (32, 32), color=(100, 200))


@pytest.fixture()
def palette_image() -> Image.Image:
    """A small palette-mode (P) image."""
    return Image.new("P", (32, 32))


# ---------------------------------------------------------------------------
# ColourStats tests
# ---------------------------------------------------------------------------


class TestColourStats:
    """Tests for the ColourStats dataclass."""

    def test_to_dict_has_required_keys(self) -> None:
        """to_dict() should include all expected keys."""
        stats = ColourStats(
            mean_r=100.0, mean_g=120.0, mean_b=80.0,
            std_r=10.0, std_g=15.0, std_b=5.0,
            dominant_colours=[(200, 100, 50), (10, 20, 30)],
        )
        d = stats.to_dict()
        expected_keys = {"mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "dominant_colours"}
        assert expected_keys.issubset(d.keys())

    def test_to_dict_dominant_colours_are_lists(self) -> None:
        """Dominant colours in to_dict() should be serialised as lists, not tuples."""
        stats = ColourStats(
            mean_r=100.0, mean_g=120.0, mean_b=80.0,
            std_r=10.0, std_g=15.0, std_b=5.0,
            dominant_colours=[(200, 100, 50)],
        )
        d = stats.to_dict()
        assert isinstance(d["dominant_colours"][0], list)

    def test_to_dict_values_rounded_to_two_decimals(self) -> None:
        """Numeric values in to_dict() should be rounded to 2 decimal places."""
        stats = ColourStats(
            mean_r=100.123456, mean_g=120.987654, mean_b=80.0,
            std_r=10.0, std_g=15.0, std_b=5.0,
            dominant_colours=[],
        )
        d = stats.to_dict()
        assert d["mean_r"] == pytest.approx(100.12)
        assert d["mean_g"] == pytest.approx(120.99)

    def test_to_dict_empty_dominant_colours(self) -> None:
        """An empty dominant_colours list should serialise as an empty list."""
        stats = ColourStats(
            mean_r=0.0, mean_g=0.0, mean_b=0.0,
            std_r=0.0, std_g=0.0, std_b=0.0,
            dominant_colours=[],
        )
        assert stats.to_dict()["dominant_colours"] == []

    def test_to_dict_multiple_dominant_colours(self) -> None:
        """Multiple dominant colours should all be serialised."""
        colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        stats = ColourStats(
            mean_r=85.0, mean_g=85.0, mean_b=85.0,
            std_r=100.0, std_g=100.0, std_b=100.0,
            dominant_colours=colours,
        )
        d = stats.to_dict()
        assert len(d["dominant_colours"]) == 3

    def test_to_dict_is_json_serialisable(self) -> None:
        """to_dict() output should be JSON-serialisable."""
        stats = ColourStats(
            mean_r=100.0, mean_g=120.0, mean_b=80.0,
            std_r=10.0, std_g=15.0, std_b=5.0,
            dominant_colours=[(200, 100, 50)],
        )
        # Should not raise.
        serialised = json.dumps(stats.to_dict())
        data = json.loads(serialised)
        assert "mean_r" in data

    def test_to_dict_mean_values_present(self) -> None:
        """Mean channel values should be present in to_dict()."""
        stats = ColourStats(
            mean_r=50.0, mean_g=100.0, mean_b=150.0,
            std_r=5.0, std_g=10.0, std_b=15.0,
            dominant_colours=[],
        )
        d = stats.to_dict()
        assert d["mean_r"] == pytest.approx(50.0)
        assert d["mean_g"] == pytest.approx(100.0)
        assert d["mean_b"] == pytest.approx(150.0)

    def test_to_dict_std_values_present(self) -> None:
        """Std deviation values should be present in to_dict()."""
        stats = ColourStats(
            mean_r=0.0, mean_g=0.0, mean_b=0.0,
            std_r=12.34, std_g=56.78, std_b=9.01,
            dominant_colours=[],
        )
        d = stats.to_dict()
        assert d["std_r"] == pytest.approx(12.34)
        assert d["std_g"] == pytest.approx(56.78)
        assert d["std_b"] == pytest.approx(9.01)

    def test_colour_values_are_three_element_lists(self) -> None:
        """Each serialised dominant colour should be a 3-element list."""
        stats = ColourStats(
            mean_r=0.0, mean_g=0.0, mean_b=0.0,
            std_r=0.0, std_g=0.0, std_b=0.0,
            dominant_colours=[(10, 20, 30), (40, 50, 60)],
        )
        d = stats.to_dict()
        for colour in d["dominant_colours"]:
            assert len(colour) == 3


# ---------------------------------------------------------------------------
# ImageReport tests
# ---------------------------------------------------------------------------


class TestImageReport:
    """Tests for the ImageReport dataclass and to_dict() serialisation."""

    def _make_report(self, **overrides) -> ImageReport:
        """Build an ImageReport with sensible defaults, optionally overriding fields."""
        defaults = dict(
            source="/tmp/img.jpg",
            is_remote=False,
            origin="/tmp",
            alt_text=None,
            extension=".jpg",
            width=100,
            height=80,
            aspect_ratio=1.25,
            file_size_bytes=1024,
            format="JPEG",
            mode="RGB",
            colour_stats=None,
            ai_score=0.72,
            ai_is_flagged=True,
            ai_threshold=0.5,
            ai_verdict="AI-generated (flagged)",
            ai_model_mode="heuristic",
            error=None,
        )
        defaults.update(overrides)
        return ImageReport(**defaults)

    def test_to_dict_returns_dict(self) -> None:
        """to_dict() should return a dictionary."""
        report = self._make_report()
        assert isinstance(report.to_dict(), dict)

    def test_to_dict_source_field(self) -> None:
        """to_dict() should include the source field."""
        report = self._make_report(source="/img/photo.jpg")
        assert report.to_dict()["source"] == "/img/photo.jpg"

    def test_to_dict_ai_score_rounded_to_four_decimals(self) -> None:
        """ai_score should be rounded to 4 decimal places in to_dict()."""
        report = self._make_report(ai_score=0.123456789)
        d = report.to_dict()
        assert d["ai_score"] == pytest.approx(0.1235)

    def test_to_dict_none_ai_score(self) -> None:
        """A None ai_score should serialise as None."""
        report = self._make_report(ai_score=None)
        assert report.to_dict()["ai_score"] is None

    def test_to_dict_colour_stats_none(self) -> None:
        """A None colour_stats should serialise as None."""
        report = self._make_report(colour_stats=None)
        assert report.to_dict()["colour_stats"] is None

    def test_to_dict_colour_stats_serialised_as_dict(self) -> None:
        """A ColourStats instance should be serialised as a dict."""
        stats = ColourStats(
            mean_r=100.0, mean_g=120.0, mean_b=80.0,
            std_r=10.0, std_g=15.0, std_b=5.0,
            dominant_colours=[(200, 100, 50)],
        )
        report = self._make_report(colour_stats=stats)
        d = report.to_dict()
        assert isinstance(d["colour_stats"], dict)
        assert "mean_r" in d["colour_stats"]

    def test_to_dict_error_field_none_on_success(self) -> None:
        """error field should be None when there is no error."""
        report = self._make_report(error=None)
        assert report.to_dict()["error"] is None

    def test_to_dict_error_field_populated(self) -> None:
        """error field should contain the error message when set."""
        report = self._make_report(error="File not found")
        assert report.to_dict()["error"] == "File not found"

    def test_to_dict_all_expected_keys_present(self) -> None:
        """to_dict() should contain all expected top-level keys."""
        report = self._make_report()
        d = report.to_dict()
        expected_keys = {
            "source", "is_remote", "origin", "alt_text", "extension",
            "width", "height", "aspect_ratio", "file_size_bytes",
            "format", "mode", "colour_stats",
            "ai_score", "ai_is_flagged", "ai_threshold",
            "ai_verdict", "ai_model_mode", "error",
        }
        assert expected_keys.issubset(d.keys())

    def test_to_dict_is_json_serialisable(self) -> None:
        """to_dict() output should be fully JSON-serialisable."""
        report = self._make_report()
        serialised = json.dumps(report.to_dict())
        data = json.loads(serialised)
        assert data["source"] == "/tmp/img.jpg"

    def test_to_dict_is_remote_field(self) -> None:
        """is_remote field should be serialised correctly."""
        report = self._make_report(is_remote=True)
        assert report.to_dict()["is_remote"] is True

    def test_to_dict_width_height(self) -> None:
        """Width and height should be serialised correctly."""
        report = self._make_report(width=1280, height=720)
        d = report.to_dict()
        assert d["width"] == 1280
        assert d["height"] == 720

    def test_to_dict_none_dimensions(self) -> None:
        """None width/height (error case) should serialise as None."""
        report = self._make_report(width=None, height=None)
        d = report.to_dict()
        assert d["width"] is None
        assert d["height"] is None

    def test_to_dict_aspect_ratio(self) -> None:
        """Aspect ratio should be serialised as a float."""
        report = self._make_report(aspect_ratio=1.7778)
        assert report.to_dict()["aspect_ratio"] == pytest.approx(1.7778)

    def test_to_dict_file_size_bytes(self) -> None:
        """file_size_bytes should be serialised correctly."""
        report = self._make_report(file_size_bytes=2048)
        assert report.to_dict()["file_size_bytes"] == 2048

    def test_to_dict_format(self) -> None:
        """format field should be serialised correctly."""
        report = self._make_report(format="PNG")
        assert report.to_dict()["format"] == "PNG"

    def test_to_dict_mode(self) -> None:
        """mode field should be serialised correctly."""
        report = self._make_report(mode="RGBA")
        assert report.to_dict()["mode"] == "RGBA"

    def test_to_dict_alt_text_none(self) -> None:
        """None alt_text should serialise as None."""
        report = self._make_report(alt_text=None)
        assert report.to_dict()["alt_text"] is None

    def test_to_dict_alt_text_value(self) -> None:
        """A non-None alt_text should serialise correctly."""
        report = self._make_report(alt_text="A sunset photo")
        assert report.to_dict()["alt_text"] == "A sunset photo"

    def test_to_dict_ai_is_flagged_true(self) -> None:
        """ai_is_flagged=True should serialise as True."""
        report = self._make_report(ai_is_flagged=True)
        assert report.to_dict()["ai_is_flagged"] is True

    def test_to_dict_ai_is_flagged_false(self) -> None:
        """ai_is_flagged=False should serialise as False."""
        report = self._make_report(ai_is_flagged=False)
        assert report.to_dict()["ai_is_flagged"] is False

    def test_to_dict_ai_threshold(self) -> None:
        """ai_threshold should serialise correctly."""
        report = self._make_report(ai_threshold=0.75)
        assert report.to_dict()["ai_threshold"] == pytest.approx(0.75)

    def test_to_dict_ai_verdict(self) -> None:
        """ai_verdict string should be serialised correctly."""
        report = self._make_report(ai_verdict="Human-made (not flagged)")
        assert report.to_dict()["ai_verdict"] == "Human-made (not flagged)"

    def test_to_dict_ai_model_mode(self) -> None:
        """ai_model_mode should be serialised correctly."""
        report = self._make_report(ai_model_mode="fine-tuned")
        assert report.to_dict()["ai_model_mode"] == "fine-tuned"

    def test_to_dict_extension(self) -> None:
        """extension field should be serialised correctly."""
        report = self._make_report(extension=".png")
        assert report.to_dict()["extension"] == ".png"

    def test_to_dict_origin(self) -> None:
        """origin field should be serialised correctly."""
        report = self._make_report(origin="/home/user/images")
        assert report.to_dict()["origin"] == "/home/user/images"


# ---------------------------------------------------------------------------
# _to_rgb_safe tests
# ---------------------------------------------------------------------------


class TestToRgbSafe:
    """Tests for the _to_rgb_safe mode conversion helper."""

    def test_rgb_passthrough(self, rgb_image: Image.Image) -> None:
        """An already-RGB image should be returned as the same object."""
        result = _to_rgb_safe(rgb_image)
        assert result.mode == "RGB"
        assert result is rgb_image

    def test_rgba_converted_to_rgb(self, rgba_image: Image.Image) -> None:
        """RGBA images should be converted to RGB."""
        result = _to_rgb_safe(rgba_image)
        assert result.mode == "RGB"
        assert result.size == rgba_image.size

    def test_grayscale_converted_to_rgb(self, grayscale_image: Image.Image) -> None:
        """Grayscale ('L') images should be converted to RGB."""
        result = _to_rgb_safe(grayscale_image)
        assert result.mode == "RGB"

    def test_la_converted_to_rgb(self, la_image: Image.Image) -> None:
        """LA (grayscale + alpha) images should be converted to RGB."""
        result = _to_rgb_safe(la_image)
        assert result.mode == "RGB"

    def test_palette_converted_to_rgb(self, palette_image: Image.Image) -> None:
        """Palette-mode ('P') images should be converted to RGB."""
        result = _to_rgb_safe(palette_image)
        assert result.mode == "RGB"

    def test_rgba_size_preserved(self, rgba_image: Image.Image) -> None:
        """Spatial dimensions should not change during RGBA->RGB conversion."""
        result = _to_rgb_safe(rgba_image)
        assert result.size == rgba_image.size

    def test_grayscale_size_preserved(self, grayscale_image: Image.Image) -> None:
        """Spatial dimensions should not change during L->RGB conversion."""
        result = _to_rgb_safe(grayscale_image)
        assert result.size == grayscale_image.size

    def test_output_has_three_channels(self, rgba_image: Image.Image) -> None:
        """The output should always have exactly 3 bands."""
        result = _to_rgb_safe(rgba_image)
        assert len(result.getbands()) == 3

    def test_cmyk_converted_to_rgb(self) -> None:
        """CMYK images should be converted to RGB."""
        cmyk = Image.new("CMYK", (32, 32), color=(0, 100, 200, 50))
        result = _to_rgb_safe(cmyk)
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# compute_colour_stats tests
# ---------------------------------------------------------------------------


class TestComputeColourStats:
    """Tests for the compute_colour_stats() function."""

    def test_returns_colour_stats_instance(self, rgb_image: Image.Image) -> None:
        """compute_colour_stats() should return a ColourStats instance."""
        stats = compute_colour_stats(rgb_image)
        assert isinstance(stats, ColourStats)

    def test_mean_values_in_valid_range(self, rgb_image: Image.Image) -> None:
        """Mean channel values should be in [0, 255]."""
        stats = compute_colour_stats(rgb_image)
        for val in (stats.mean_r, stats.mean_g, stats.mean_b):
            assert 0.0 <= val <= 255.0

    def test_std_values_non_negative(self, rgb_image: Image.Image) -> None:
        """Standard deviation values should be >= 0."""
        stats = compute_colour_stats(rgb_image)
        for val in (stats.std_r, stats.std_g, stats.std_b):
            assert val >= 0.0

    def test_solid_red_mean_accurate(self) -> None:
        """A solid red image should have mean_r ~= 255, mean_g ~= 0, mean_b ~= 0."""
        img = Image.new("RGB", (50, 50), color=(255, 0, 0))
        stats = compute_colour_stats(img)
        assert stats.mean_r == pytest.approx(255.0)
        assert stats.mean_g == pytest.approx(0.0)
        assert stats.mean_b == pytest.approx(0.0)

    def test_solid_colour_std_is_zero(self) -> None:
        """A solid-colour image should have zero standard deviation for all channels."""
        img = Image.new("RGB", (50, 50), color=(100, 200, 50))
        stats = compute_colour_stats(img)
        assert stats.std_r == pytest.approx(0.0, abs=1e-3)
        assert stats.std_g == pytest.approx(0.0, abs=1e-3)
        assert stats.std_b == pytest.approx(0.0, abs=1e-3)

    def test_dominant_colours_count_respects_n_colours(self, rgb_image: Image.Image) -> None:
        """Number of dominant colours should not exceed n_colours."""
        stats = compute_colour_stats(rgb_image, n_colours=3)
        assert len(stats.dominant_colours) <= 3

    def test_dominant_colours_are_rgb_tuples(self, rgb_image: Image.Image) -> None:
        """Each dominant colour should be a 3-element tuple/sequence."""
        stats = compute_colour_stats(rgb_image)
        for colour in stats.dominant_colours:
            assert len(colour) == 3
            for c in colour:
                assert 0 <= c <= 255

    def test_rgba_image_processed(self, rgba_image: Image.Image) -> None:
        """RGBA images should be processed without error."""
        stats = compute_colour_stats(rgba_image)
        assert isinstance(stats, ColourStats)

    def test_grayscale_image_processed(self, grayscale_image: Image.Image) -> None:
        """Grayscale images should be processed without error."""
        stats = compute_colour_stats(grayscale_image)
        assert isinstance(stats, ColourStats)

    def test_tiny_image_processed(self, tiny_image: Image.Image) -> None:
        """A 1x1 pixel image should be processed without error."""
        stats = compute_colour_stats(tiny_image)
        assert isinstance(stats, ColourStats)

    def test_n_colours_one(self, rgb_image: Image.Image) -> None:
        """n_colours=1 should return at most one dominant colour."""
        stats = compute_colour_stats(rgb_image, n_colours=1)
        assert len(stats.dominant_colours) <= 1

    def test_invalid_n_colours_zero_raises(self, rgb_image: Image.Image) -> None:
        """n_colours=0 should raise ValueError."""
        with pytest.raises(ValueError, match="n_colours"):
            compute_colour_stats(rgb_image, n_colours=0)

    def test_invalid_n_colours_negative_raises(self, rgb_image: Image.Image) -> None:
        """A negative n_colours should raise ValueError."""
        with pytest.raises(ValueError, match="n_colours"):
            compute_colour_stats(rgb_image, n_colours=-1)

    def test_non_pil_raises_type_error(self) -> None:
        """Passing a non-PIL object should raise TypeError."""
        with pytest.raises(TypeError):
            compute_colour_stats("not an image")  # type: ignore[arg-type]

    def test_non_pil_array_raises_type_error(self) -> None:
        """Passing a numpy array should raise TypeError."""
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(TypeError):
            compute_colour_stats(arr)  # type: ignore[arg-type]

    def test_default_palette_colours_count(self, rgb_image: Image.Image) -> None:
        """Default n_colours should yield at most DEFAULT_PALETTE_COLOURS dominant colours."""
        stats = compute_colour_stats(rgb_image)
        assert len(stats.dominant_colours) <= DEFAULT_PALETTE_COLOURS

    def test_la_image_processed(self, la_image: Image.Image) -> None:
        """LA (grayscale+alpha) images should be processed without error."""
        stats = compute_colour_stats(la_image)
        assert isinstance(stats, ColourStats)

    def test_palette_image_processed(self, palette_image: Image.Image) -> None:
        """Palette-mode images should be processed without error."""
        stats = compute_colour_stats(palette_image)
        assert isinstance(stats, ColourStats)

    def test_wide_image_processed(self, wide_image: Image.Image) -> None:
        """A wide panoramic image should be processed without error."""
        stats = compute_colour_stats(wide_image)
        assert isinstance(stats, ColourStats)

    def test_mean_values_are_floats(self, rgb_image: Image.Image) -> None:
        """Mean values should be Python floats."""
        stats = compute_colour_stats(rgb_image)
        assert isinstance(stats.mean_r, float)
        assert isinstance(stats.mean_g, float)
        assert isinstance(stats.mean_b, float)

    def test_std_values_are_floats(self, rgb_image: Image.Image) -> None:
        """Std values should be Python floats."""
        stats = compute_colour_stats(rgb_image)
        assert isinstance(stats.std_r, float)
        assert isinstance(stats.std_g, float)
        assert isinstance(stats.std_b, float)

    def test_solid_blue_mean_accurate(self) -> None:
        """A solid blue image should have mean_r ~= 0, mean_g ~= 0, mean_b ~= 255."""
        img = Image.new("RGB", (50, 50), color=(0, 0, 255))
        stats = compute_colour_stats(img)
        assert stats.mean_r == pytest.approx(0.0)
        assert stats.mean_g == pytest.approx(0.0)
        assert stats.mean_b == pytest.approx(255.0)

    def test_custom_n_colours_five(self, rgb_image: Image.Image) -> None:
        """n_colours=5 should yield at most 5 dominant colours."""
        stats = compute_colour_stats(rgb_image, n_colours=5)
        assert len(stats.dominant_colours) <= 5


# ---------------------------------------------------------------------------
# _extract_dominant_colours tests
# ---------------------------------------------------------------------------


class TestExtractDominantColours:
    """Tests for the _extract_dominant_colours helper."""

    def test_basic_extraction_returns_list(self) -> None:
        """Should return a list of colour tuples."""
        img = Image.new("RGB", (50, 50), color=(200, 100, 50))
        colours = _extract_dominant_colours(img, n_colours=3)
        assert isinstance(colours, list)

    def test_count_does_not_exceed_n_colours(self) -> None:
        """Number of returned colours should not exceed n_colours."""
        img = Image.new("RGB", (50, 50), color=(200, 100, 50))
        colours = _extract_dominant_colours(img, n_colours=3)
        assert len(colours) <= 3

    def test_each_colour_is_three_element_sequence(self) -> None:
        """Each colour should be a 3-element sequence."""
        img = Image.new("RGB", (50, 50), color=(10, 20, 30))
        colours = _extract_dominant_colours(img, n_colours=5)
        for c in colours:
            assert len(c) == 3

    def test_values_in_byte_range(self) -> None:
        """All channel values should be in [0, 255]."""
        img = Image.new("RGB", (64, 64), color=(128, 64, 192))
        colours = _extract_dominant_colours(img, n_colours=5)
        for r, g, b in colours:
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_tiny_image_does_not_crash(self) -> None:
        """A 1x1 pixel image should not crash the extraction."""
        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        colours = _extract_dominant_colours(img, n_colours=3)
        assert isinstance(colours, list)

    def test_tiny_image_returns_at_least_one_colour(self) -> None:
        """A 1x1 pixel image should return at least one colour."""
        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        colours = _extract_dominant_colours(img, n_colours=3)
        assert len(colours) >= 1

    def test_multicolour_image_returns_colours(self) -> None:
        """A checkerboard image should return at least one colour cluster."""
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        pixels = img.load()
        assert pixels is not None
        for y in range(100):
            for x in range(100):
                if (x + y) % 2 == 0:
                    pixels[x, y] = (255, 255, 255)
                else:
                    pixels[x, y] = (0, 0, 0)
        colours = _extract_dominant_colours(img, n_colours=5)
        assert len(colours) >= 1

    def test_n_colours_one_returns_one(self) -> None:
        """n_colours=1 should return at most one colour."""
        img = Image.new("RGB", (50, 50), color=(200, 100, 50))
        colours = _extract_dominant_colours(img, n_colours=1)
        assert len(colours) <= 1

    def test_large_image_does_not_crash(self) -> None:
        """A larger image should be processed without error."""
        img = Image.new("RGB", (512, 512), color=(128, 64, 32))
        colours = _extract_dominant_colours(img, n_colours=5)
        assert isinstance(colours, list)
        assert len(colours) >= 1


# ---------------------------------------------------------------------------
# _get_file_size tests
# ---------------------------------------------------------------------------


class TestGetFileSize:
    """Tests for the _get_file_size helper."""

    def test_local_file_returns_correct_size(self, tmp_path: Path) -> None:
        """Size of an existing local file should be returned correctly."""
        f = tmp_path / "img.jpg"
        data = b"fake jpeg data" * 100
        f.write_bytes(data)
        ref = ImageRef(
            source=str(f),
            is_remote=False,
            origin=str(tmp_path),
        )
        size = _get_file_size(ref)
        assert size == len(data)

    def test_remote_ref_returns_none(self, remote_ref: ImageRef) -> None:
        """A remote ImageRef should always return None."""
        assert _get_file_size(remote_ref) is None

    def test_missing_local_file_returns_none(self, tmp_path: Path) -> None:
        """A missing local file should return None instead of raising."""
        ref = ImageRef(
            source=str(tmp_path / "nonexistent.jpg"),
            is_remote=False,
            origin=str(tmp_path),
        )
        assert _get_file_size(ref) is None

    def test_local_empty_file_returns_zero(self, tmp_path: Path) -> None:
        """An empty local file should return size 0."""
        f = tmp_path / "empty.jpg"
        f.write_bytes(b"")
        ref = ImageRef(source=str(f), is_remote=False, origin=str(tmp_path))
        assert _get_file_size(ref) == 0

    def test_size_matches_os_stat(self, tmp_path: Path) -> None:
        """Returned size should match os.path.getsize()."""
        f = tmp_path / "test.png"
        data = b"x" * 1234
        f.write_bytes(data)
        ref = ImageRef(source=str(f), is_remote=False, origin=str(tmp_path))
        assert _get_file_size(ref) == os.path.getsize(str(f))


# ---------------------------------------------------------------------------
# generate_report tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for the generate_report() function."""

    def test_returns_image_report_instance(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """generate_report() should return an ImageReport instance."""
        report = generate_report(local_ref, classification, rgb_image)
        assert isinstance(report, ImageReport)

    def test_width_and_height_from_image(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """Width and height should match the PIL image dimensions."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.width == 100
        assert report.height == 80

    def test_aspect_ratio_computed_correctly(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """Aspect ratio should be width / height."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.aspect_ratio == pytest.approx(100 / 80, rel=1e-3)

    def test_ai_score_from_classification(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """AI score should come from the classification result."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.ai_score == pytest.approx(0.72)

    def test_ai_is_flagged_true(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """ai_is_flagged should match the classification result."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.ai_is_flagged is True

    def test_ai_threshold_from_classification(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """ai_threshold should come from the classification result."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.ai_threshold == pytest.approx(0.5)

    def test_ai_verdict_from_classification(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """ai_verdict should come from the classification result."""
        report = generate_report(local_ref, classification, rgb_image)
        assert "AI-generated" in report.ai_verdict

    def test_source_matches_ref(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """source should match the ImageRef source."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.source == local_ref.source

    def test_is_remote_matches_ref(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """is_remote should be False for a local ref."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.is_remote is False

    def test_origin_matches_ref(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """origin should match the ImageRef origin."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.origin == local_ref.origin

    def test_extension_matches_ref(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """extension should match the ImageRef extension."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.extension == ".jpg"

    def test_colour_stats_present_and_valid(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """colour_stats should be populated with a ColourStats instance."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.colour_stats is not None
        assert isinstance(report.colour_stats, ColourStats)

    def test_error_is_none_on_success(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """error should be None when generation succeeds."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.error is None

    def test_file_size_positive_for_existing_local_file(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """file_size_bytes should be a positive integer for an existing local file."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.file_size_bytes is not None
        assert report.file_size_bytes > 0

    def test_file_size_none_for_remote(
        self, remote_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """file_size_bytes should be None for remote images."""
        report = generate_report(remote_ref, classification, rgb_image)
        assert report.file_size_bytes is None

    def test_rgba_image_handled(
        self, local_ref: ImageRef, classification: ClassificationResult, rgba_image: Image.Image
    ) -> None:
        """RGBA images should be processed without error."""
        report = generate_report(local_ref, classification, rgba_image)
        assert isinstance(report, ImageReport)
        assert report.width == 64

    def test_grayscale_image_handled(
        self, local_ref: ImageRef, classification: ClassificationResult, grayscale_image: Image.Image
    ) -> None:
        """Grayscale images should be processed without error."""
        report = generate_report(local_ref, classification, grayscale_image)
        assert isinstance(report, ImageReport)

    def test_tiny_image_handled(
        self, local_ref: ImageRef, classification: ClassificationResult, tiny_image: Image.Image
    ) -> None:
        """A 1x1 pixel image should be processed without error."""
        report = generate_report(local_ref, classification, tiny_image)
        assert report.width == 1
        assert report.height == 1
        assert report.aspect_ratio == pytest.approx(1.0)

    def test_wide_image_aspect_ratio(
        self, local_ref: ImageRef, classification: ClassificationResult, wide_image: Image.Image
    ) -> None:
        """Wide image aspect ratio should be computed correctly."""
        report = generate_report(local_ref, classification, wide_image)
        assert report.aspect_ratio == pytest.approx(1920 / 400, rel=1e-3)

    def test_type_error_for_non_pil_image(
        self, local_ref: ImageRef, classification: ClassificationResult
    ) -> None:
        """Passing a non-PIL image should raise TypeError."""
        with pytest.raises(TypeError, match="PIL.Image.Image"):
            generate_report(local_ref, classification, "not an image")  # type: ignore[arg-type]

    def test_type_error_for_non_classification(
        self, local_ref: ImageRef, rgb_image: Image.Image
    ) -> None:
        """Passing a non-ClassificationResult should raise TypeError."""
        with pytest.raises(TypeError, match="ClassificationResult"):
            generate_report(local_ref, "not a result", rgb_image)  # type: ignore[arg-type]

    def test_custom_palette_colours(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """A custom palette_colours argument should be respected."""
        report = generate_report(
            local_ref, classification, rgb_image, palette_colours=3
        )
        assert report.colour_stats is not None
        assert len(report.colour_stats.dominant_colours) <= 3

    def test_alt_text_preserved_from_remote_ref(
        self, remote_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """alt_text from the ref should be preserved in the report."""
        report = generate_report(remote_ref, classification, rgb_image)
        assert report.alt_text == "A beautiful sunset"

    def test_to_dict_is_json_compatible(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """to_dict() output should be fully JSON-serialisable."""
        report = generate_report(local_ref, classification, rgb_image)
        serialised = json.dumps(report.to_dict())
        data = json.loads(serialised)
        assert data["ai_score"] is not None

    def test_model_mode_in_report(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """ai_model_mode should be taken from the classification result."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.ai_model_mode == "heuristic"

    def test_remote_ref_is_remote_true(
        self, remote_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """is_remote should be True for a remote ref."""
        report = generate_report(remote_ref, classification, rgb_image)
        assert report.is_remote is True

    def test_non_flagged_classification(
        self, local_ref: ImageRef, rgb_image: Image.Image
    ) -> None:
        """A non-flagged classification should produce is_flagged=False in the report."""
        safe_cls = ClassificationResult(
            score=0.2,
            is_flagged=False,
            threshold=0.5,
            verdict="Human-made (not flagged)",
            model_mode="heuristic",
        )
        report = generate_report(local_ref, safe_cls, rgb_image)
        assert report.ai_is_flagged is False
        assert "Human-made" in report.ai_verdict

    def test_image_mode_in_report(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """The image mode should be captured in the report."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.mode == "RGB"

    def test_image_format_unknown_for_in_memory(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """An in-memory image (not loaded from file) may have format=None -> 'UNKNOWN'."""
        # rgb_image fixture is Image.new(), so format is None -> stored as 'UNKNOWN'
        report = generate_report(local_ref, classification, rgb_image)
        # format can be None (from PIL) stored as "UNKNOWN" or the actual format
        assert report.format is not None

    def test_colour_stats_mean_in_range(
        self, local_ref: ImageRef, classification: ClassificationResult, rgb_image: Image.Image
    ) -> None:
        """Colour stats mean values should be in [0, 255]."""
        report = generate_report(local_ref, classification, rgb_image)
        assert report.colour_stats is not None
        for val in (
            report.colour_stats.mean_r,
            report.colour_stats.mean_g,
            report.colour_stats.mean_b,
        ):
            assert 0.0 <= val <= 255.0

    def test_non_pil_int_raises_type_error(
        self, local_ref: ImageRef, classification: ClassificationResult
    ) -> None:
        """Passing an integer as the image should raise TypeError."""
        with pytest.raises(TypeError):
            generate_report(local_ref, classification, 42)  # type: ignore[arg-type]

    def test_non_classification_int_raises_type_error(
        self, local_ref: ImageRef, rgb_image: Image.Image
    ) -> None:
        """Passing an integer as classification should raise TypeError."""
        with pytest.raises(TypeError):
            generate_report(local_ref, 99, rgb_image)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# generate_report_from_path tests
# ---------------------------------------------------------------------------


class TestGenerateReportFromPath:
    """Tests for the generate_report_from_path() convenience function."""

    def test_valid_jpeg_returns_report(
        self, valid_jpeg_path: Path, classification: ClassificationResult
    ) -> None:
        """A valid JPEG file should produce a report with no error."""
        ref = ImageRef(
            source=str(valid_jpeg_path),
            is_remote=False,
            origin=str(valid_jpeg_path.parent),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        assert isinstance(report, ImageReport)
        assert report.error is None
        assert report.width is not None

    def test_valid_jpeg_width_and_height(
        self, valid_jpeg_path: Path, classification: ClassificationResult
    ) -> None:
        """Width and height should be populated for a valid JPEG."""
        ref = ImageRef(
            source=str(valid_jpeg_path),
            is_remote=False,
            origin=str(valid_jpeg_path.parent),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        assert report.width == 64
        assert report.height == 64

    def test_missing_file_returns_error_report(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """A missing file should return an error report instead of raising."""
        ref = ImageRef(
            source=str(tmp_path / "missing.jpg"),
            is_remote=False,
            origin=str(tmp_path),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        assert isinstance(report, ImageReport)
        assert report.error is not None
        assert "not found" in report.error.lower()

    def test_corrupt_file_returns_error_report(
        self, corrupt_file_path: Path, classification: ClassificationResult
    ) -> None:
        """A corrupt file should return an error report instead of raising."""
        ref = ImageRef(
            source=str(corrupt_file_path),
            is_remote=False,
            origin=str(corrupt_file_path.parent),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        assert isinstance(report, ImageReport)
        assert report.error is not None

    def test_error_report_preserves_source(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """An error report should still preserve the source field."""
        missing = str(tmp_path / "does_not_exist.png")
        ref = ImageRef(
            source=missing,
            is_remote=False,
            origin=str(tmp_path),
            extension=".png",
        )
        report = generate_report_from_path(ref, classification)
        assert report.source == missing

    def test_error_report_preserves_ai_score(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """An error report should preserve AI score from the classification result."""
        ref = ImageRef(
            source=str(tmp_path / "missing.jpg"),
            is_remote=False,
            origin=str(tmp_path),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        assert report.ai_score == pytest.approx(0.72)
        assert report.ai_is_flagged is True

    def test_error_report_width_is_none(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """An error report should have None for width, height, and colour_stats."""
        ref = ImageRef(
            source=str(tmp_path / "nope.jpg"),
            is_remote=False,
            origin=str(tmp_path),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        assert report.width is None
        assert report.height is None
        assert report.colour_stats is None

    def test_png_file_format_is_png(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """A PNG file should produce a report with format='PNG'."""
        p = tmp_path / "img.png"
        Image.new("RGB", (32, 32), color=(0, 255, 0)).save(str(p), format="PNG")
        ref = ImageRef(
            source=str(p),
            is_remote=False,
            origin=str(tmp_path),
            extension=".png",
        )
        report = generate_report_from_path(ref, classification)
        assert report.error is None
        assert report.format == "PNG"

    def test_result_is_always_image_report(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """Even on error, the result should always be an ImageReport."""
        ref = ImageRef(
            source=str(tmp_path / "nonexistent.bmp"),
            is_remote=False,
            origin=str(tmp_path),
            extension=".bmp",
        )
        report = generate_report_from_path(ref, classification)
        assert isinstance(report, ImageReport)

    def test_error_report_preserves_is_remote(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """Error report should preserve the is_remote field from the ref."""
        ref = ImageRef(
            source=str(tmp_path / "missing.jpg"),
            is_remote=False,
            origin=str(tmp_path),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        assert report.is_remote is False

    def test_error_report_preserves_origin(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """Error report should preserve the origin field from the ref."""
        ref = ImageRef(
            source=str(tmp_path / "missing.jpg"),
            is_remote=False,
            origin=str(tmp_path),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        assert report.origin == str(tmp_path)

    def test_error_report_preserves_extension(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """Error report should preserve the extension field from the ref."""
        ref = ImageRef(
            source=str(tmp_path / "missing.tiff"),
            is_remote=False,
            origin=str(tmp_path),
            extension=".tiff",
        )
        report = generate_report_from_path(ref, classification)
        assert report.extension == ".tiff"

    def test_error_report_preserves_ai_threshold(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """Error report should preserve ai_threshold from the classification."""
        ref = ImageRef(
            source=str(tmp_path / "missing.jpg"),
            is_remote=False,
            origin=str(tmp_path),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        assert report.ai_threshold == pytest.approx(0.5)

    def test_valid_png_colour_stats_present(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """A valid PNG should produce a report with colour_stats populated."""
        p = tmp_path / "coloured.png"
        Image.new("RGB", (64, 64), color=(128, 64, 32)).save(str(p), format="PNG")
        ref = ImageRef(
            source=str(p),
            is_remote=False,
            origin=str(tmp_path),
            extension=".png",
        )
        report = generate_report_from_path(ref, classification)
        assert report.colour_stats is not None
        assert isinstance(report.colour_stats, ColourStats)

    def test_error_report_to_dict_is_json_serialisable(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
        """An error report's to_dict() should be JSON-serialisable."""
        ref = ImageRef(
            source=str(tmp_path / "missing.jpg"),
            is_remote=False,
            origin=str(tmp_path),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        serialised = json.dumps(report.to_dict())
        data = json.loads(serialised)
        assert data["error"] is not None
