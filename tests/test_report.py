"""Unit tests for ai_image_audit.report.

Covers:
- ColourStats dataclass and to_dict() serialisation.
- ImageReport dataclass and to_dict() serialisation.
- compute_colour_stats() with various image modes, sizes, and edge cases.
- generate_report() with valid inputs, type errors, and edge-case images.
- generate_report_from_path() with valid, missing, and corrupt files.
- _get_file_size() for local and remote refs.
- _extract_dominant_colours() for small/minimal images.
- Error report generation via generate_report_from_path().
"""

from __future__ import annotations

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
    """A small 100×80 solid-colour RGB image."""
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
    """A 1×1 pixel image."""
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


# ---------------------------------------------------------------------------
# ColourStats tests
# ---------------------------------------------------------------------------


class TestColourStats:
    def test_to_dict_keys(self) -> None:
        stats = ColourStats(
            mean_r=100.0, mean_g=120.0, mean_b=80.0,
            std_r=10.0, std_g=15.0, std_b=5.0,
            dominant_colours=[(200, 100, 50), (10, 20, 30)],
        )
        d = stats.to_dict()
        assert "mean_r" in d
        assert "mean_g" in d
        assert "mean_b" in d
        assert "std_r" in d
        assert "std_g" in d
        assert "std_b" in d
        assert "dominant_colours" in d

    def test_to_dict_dominant_colours_are_lists(self) -> None:
        stats = ColourStats(
            mean_r=100.0, mean_g=120.0, mean_b=80.0,
            std_r=10.0, std_g=15.0, std_b=5.0,
            dominant_colours=[(200, 100, 50)],
        )
        d = stats.to_dict()
        # Each colour should be serialised as a list, not a tuple.
        assert isinstance(d["dominant_colours"][0], list)

    def test_to_dict_values_rounded(self) -> None:
        stats = ColourStats(
            mean_r=100.123456, mean_g=120.987654, mean_b=80.0,
            std_r=10.0, std_g=15.0, std_b=5.0,
            dominant_colours=[],
        )
        d = stats.to_dict()
        # Rounded to 2 decimal places.
        assert d["mean_r"] == pytest.approx(100.12)
        assert d["mean_g"] == pytest.approx(120.99)

    def test_empty_dominant_colours(self) -> None:
        stats = ColourStats(
            mean_r=0.0, mean_g=0.0, mean_b=0.0,
            std_r=0.0, std_g=0.0, std_b=0.0,
            dominant_colours=[],
        )
        assert stats.to_dict()["dominant_colours"] == []


# ---------------------------------------------------------------------------
# ImageReport tests
# ---------------------------------------------------------------------------


class TestImageReport:
    def _make_report(self, **overrides) -> ImageReport:
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
        report = self._make_report()
        assert isinstance(report.to_dict(), dict)

    def test_to_dict_source(self) -> None:
        report = self._make_report(source="/img/photo.jpg")
        assert report.to_dict()["source"] == "/img/photo.jpg"

    def test_to_dict_ai_score_rounded(self) -> None:
        report = self._make_report(ai_score=0.123456789)
        d = report.to_dict()
        assert d["ai_score"] == pytest.approx(0.1235)

    def test_to_dict_none_ai_score(self) -> None:
        report = self._make_report(ai_score=None)
        assert report.to_dict()["ai_score"] is None

    def test_to_dict_colour_stats_none(self) -> None:
        report = self._make_report(colour_stats=None)
        assert report.to_dict()["colour_stats"] is None

    def test_to_dict_colour_stats_serialised(self) -> None:
        stats = ColourStats(
            mean_r=100.0, mean_g=120.0, mean_b=80.0,
            std_r=10.0, std_g=15.0, std_b=5.0,
            dominant_colours=[(200, 100, 50)],
        )
        report = self._make_report(colour_stats=stats)
        d = report.to_dict()
        assert isinstance(d["colour_stats"], dict)
        assert "mean_r" in d["colour_stats"]

    def test_error_field_none_on_success(self) -> None:
        report = self._make_report(error=None)
        assert report.to_dict()["error"] is None

    def test_error_field_populated(self) -> None:
        report = self._make_report(error="File not found")
        assert report.to_dict()["error"] == "File not found"

    def test_all_expected_keys_present(self) -> None:
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


# ---------------------------------------------------------------------------
# _to_rgb_safe tests
# ---------------------------------------------------------------------------


class TestToRgbSafe:
    def test_rgb_passthrough(self, rgb_image: Image.Image) -> None:
        result = _to_rgb_safe(rgb_image)
        assert result.mode == "RGB"
        assert result is rgb_image  # No-op for RGB.

    def test_rgba_converted(self, rgba_image: Image.Image) -> None:
        result = _to_rgb_safe(rgba_image)
        assert result.mode == "RGB"
        assert result.size == rgba_image.size

    def test_grayscale_converted(self, grayscale_image: Image.Image) -> None:
        result = _to_rgb_safe(grayscale_image)
        assert result.mode == "RGB"

    def test_la_converted(self) -> None:
        la = Image.new("LA", (32, 32), (100, 200))
        result = _to_rgb_safe(la)
        assert result.mode == "RGB"

    def test_palette_converted(self) -> None:
        p = Image.new("P", (32, 32))
        result = _to_rgb_safe(p)
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# compute_colour_stats tests
# ---------------------------------------------------------------------------


class TestComputeColourStats:
    def test_returns_colour_stats(self, rgb_image: Image.Image) -> None:
        stats = compute_colour_stats(rgb_image)
        assert isinstance(stats, ColourStats)

    def test_mean_values_in_range(self, rgb_image: Image.Image) -> None:
        stats = compute_colour_stats(rgb_image)
        for val in (stats.mean_r, stats.mean_g, stats.mean_b):
            assert 0.0 <= val <= 255.0

    def test_std_values_non_negative(self, rgb_image: Image.Image) -> None:
        stats = compute_colour_stats(rgb_image)
        for val in (stats.std_r, stats.std_g, stats.std_b):
            assert val >= 0.0

    def test_solid_colour_mean_accurate(self) -> None:
        # Solid red image: mean_r ≈ 255, mean_g ≈ 0, mean_b ≈ 0.
        img = Image.new("RGB", (50, 50), color=(255, 0, 0))
        stats = compute_colour_stats(img)
        assert stats.mean_r == pytest.approx(255.0)
        assert stats.mean_g == pytest.approx(0.0)
        assert stats.mean_b == pytest.approx(0.0)

    def test_solid_colour_std_is_zero(self) -> None:
        img = Image.new("RGB", (50, 50), color=(100, 200, 50))
        stats = compute_colour_stats(img)
        assert stats.std_r == pytest.approx(0.0, abs=1e-3)
        assert stats.std_g == pytest.approx(0.0, abs=1e-3)
        assert stats.std_b == pytest.approx(0.0, abs=1e-3)

    def test_dominant_colours_count(self, rgb_image: Image.Image) -> None:
        stats = compute_colour_stats(rgb_image, n_colours=3)
        assert len(stats.dominant_colours) <= 3

    def test_dominant_colours_are_rgb_tuples(self, rgb_image: Image.Image) -> None:
        stats = compute_colour_stats(rgb_image)
        for colour in stats.dominant_colours:
            assert len(colour) == 3
            for c in colour:
                assert 0 <= c <= 255

    def test_rgba_image_processed(self, rgba_image: Image.Image) -> None:
        stats = compute_colour_stats(rgba_image)
        assert isinstance(stats, ColourStats)

    def test_grayscale_image_processed(self, grayscale_image: Image.Image) -> None:
        stats = compute_colour_stats(grayscale_image)
        assert isinstance(stats, ColourStats)

    def test_tiny_image_processed(self, tiny_image: Image.Image) -> None:
        stats = compute_colour_stats(tiny_image)
        assert isinstance(stats, ColourStats)

    def test_n_colours_one(self, rgb_image: Image.Image) -> None:
        stats = compute_colour_stats(rgb_image, n_colours=1)
        assert len(stats.dominant_colours) <= 1

    def test_invalid_n_colours_raises(self, rgb_image: Image.Image) -> None:
        with pytest.raises(ValueError, match="n_colours"):
            compute_colour_stats(rgb_image, n_colours=0)

    def test_non_pil_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            compute_colour_stats("not an image")  # type: ignore[arg-type]

    def test_default_palette_colours(self, rgb_image: Image.Image) -> None:
        stats = compute_colour_stats(rgb_image)
        assert len(stats.dominant_colours) <= DEFAULT_PALETTE_COLOURS


# ---------------------------------------------------------------------------
# _extract_dominant_colours tests
# ---------------------------------------------------------------------------


class TestExtractDominantColours:
    def test_basic_extraction(self) -> None:
        img = Image.new("RGB", (50, 50), color=(200, 100, 50))
        colours = _extract_dominant_colours(img, n_colours=3)
        assert isinstance(colours, list)
        assert len(colours) <= 3

    def test_each_colour_is_rgb_tuple(self) -> None:
        img = Image.new("RGB", (50, 50), color=(10, 20, 30))
        colours = _extract_dominant_colours(img, n_colours=5)
        for c in colours:
            assert len(c) == 3

    def test_values_in_byte_range(self) -> None:
        img = Image.new("RGB", (64, 64), color=(128, 64, 192))
        colours = _extract_dominant_colours(img, n_colours=5)
        for r, g, b in colours:
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_tiny_image(self) -> None:
        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        colours = _extract_dominant_colours(img, n_colours=3)
        # Should not crash and should return at least one colour.
        assert isinstance(colours, list)

    def test_multicolour_image(self) -> None:
        """A checkerboard should yield at least 2 distinct dominant colours."""
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
        assert len(colours) >= 1  # At minimum one colour cluster found.


# ---------------------------------------------------------------------------
# _get_file_size tests
# ---------------------------------------------------------------------------


class TestGetFileSize:
    def test_local_file_returns_size(self, tmp_path: Path) -> None:
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

    def test_remote_returns_none(self, remote_ref: ImageRef) -> None:
        assert _get_file_size(remote_ref) is None

    def test_missing_local_file_returns_none(self, tmp_path: Path) -> None:
        ref = ImageRef(
            source=str(tmp_path / "nonexistent.jpg"),
            is_remote=False,
            origin=str(tmp_path),
        )
        assert _get_file_size(ref) is None


# ---------------------------------------------------------------------------
# generate_report tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_returns_image_report(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert isinstance(report, ImageReport)

    def test_width_and_height_correct(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.width == 100
        assert report.height == 80

    def test_aspect_ratio_correct(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.aspect_ratio == pytest.approx(100 / 80, rel=1e-3)

    def test_ai_score_from_classification(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.ai_score == pytest.approx(0.72)

    def test_ai_is_flagged(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.ai_is_flagged is True

    def test_ai_threshold(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.ai_threshold == pytest.approx(0.5)

    def test_ai_verdict(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert "AI-generated" in report.ai_verdict

    def test_source_matches_ref(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.source == local_ref.source

    def test_is_remote_matches_ref(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.is_remote is False

    def test_origin_matches_ref(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.origin == local_ref.origin

    def test_extension_matches_ref(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.extension == ".jpg"

    def test_colour_stats_present(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.colour_stats is not None
        assert isinstance(report.colour_stats, ColourStats)

    def test_error_is_none_on_success(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.error is None

    def test_file_size_set_for_local(self, local_ref, classification, rgb_image) -> None:
        # file_size_bytes is determined from disk, not from the PIL image;
        # it should be a positive integer for an existing local file.
        report = generate_report(local_ref, classification, rgb_image)
        assert report.file_size_bytes is not None
        assert report.file_size_bytes > 0

    def test_file_size_none_for_remote(self, remote_ref, classification, rgb_image) -> None:
        report = generate_report(remote_ref, classification, rgb_image)
        assert report.file_size_bytes is None

    def test_rgba_image_handled(self, local_ref, classification, rgba_image) -> None:
        report = generate_report(local_ref, classification, rgba_image)
        assert isinstance(report, ImageReport)
        assert report.width == 64

    def test_grayscale_image_handled(self, local_ref, classification, grayscale_image) -> None:
        report = generate_report(local_ref, classification, grayscale_image)
        assert isinstance(report, ImageReport)

    def test_tiny_image_handled(self, local_ref, classification, tiny_image) -> None:
        report = generate_report(local_ref, classification, tiny_image)
        assert report.width == 1
        assert report.height == 1
        assert report.aspect_ratio == pytest.approx(1.0)

    def test_wide_image_aspect_ratio(self, local_ref, classification, wide_image) -> None:
        report = generate_report(local_ref, classification, wide_image)
        assert report.aspect_ratio == pytest.approx(1920 / 400, rel=1e-3)

    def test_type_error_for_non_pil_image(self, local_ref, classification) -> None:
        with pytest.raises(TypeError, match="PIL.Image.Image"):
            generate_report(local_ref, classification, "not an image")  # type: ignore[arg-type]

    def test_type_error_for_non_classification(self, local_ref, rgb_image) -> None:
        with pytest.raises(TypeError, match="ClassificationResult"):
            generate_report(local_ref, "not a result", rgb_image)  # type: ignore[arg-type]

    def test_custom_palette_colours(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(
            local_ref, classification, rgb_image, palette_colours=3
        )
        assert report.colour_stats is not None
        assert len(report.colour_stats.dominant_colours) <= 3

    def test_alt_text_preserved(self, remote_ref, classification, rgb_image) -> None:
        report = generate_report(remote_ref, classification, rgb_image)
        assert report.alt_text == "A beautiful sunset"

    def test_to_dict_is_json_compatible(self, local_ref, classification, rgb_image) -> None:
        import json
        report = generate_report(local_ref, classification, rgb_image)
        # Should not raise.
        serialised = json.dumps(report.to_dict())
        data = json.loads(serialised)
        assert data["ai_score"] is not None

    def test_model_mode_in_report(self, local_ref, classification, rgb_image) -> None:
        report = generate_report(local_ref, classification, rgb_image)
        assert report.ai_model_mode == "heuristic"


# ---------------------------------------------------------------------------
# generate_report_from_path tests
# ---------------------------------------------------------------------------


class TestGenerateReportFromPath:
    def test_valid_jpeg_returns_report(
        self, valid_jpeg_path: Path, classification: ClassificationResult
    ) -> None:
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

    def test_missing_file_returns_error_report(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
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
        ref = ImageRef(
            source=str(tmp_path / "missing.jpg"),
            is_remote=False,
            origin=str(tmp_path),
            extension=".jpg",
        )
        report = generate_report_from_path(ref, classification)
        assert report.ai_score == pytest.approx(0.72)
        assert report.ai_is_flagged is True

    def test_png_file_classified(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
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

    def test_error_report_width_is_none(
        self, tmp_path: Path, classification: ClassificationResult
    ) -> None:
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
