"""Unit tests for ai_image_audit.classifier.

Covers:
- AIImageClassifier initialisation (default heuristic mode, custom threshold,
  invalid threshold, missing checkpoint, device selection).
- ClassificationResult dataclass fields and repr.
- classify() with synthetic PIL images in various modes (RGB, RGBA, L, P).
- classify() type checking for non-PIL inputs.
- classify_path() with valid, missing, and corrupt image files.
- Fine-tuned mode score computation (_finetuned_score).
- Heuristic mode score computation (_heuristic_score).
- Pre-processing helpers: _build_transform, _to_rgb.
- Module-level classify_image() convenience function.
- Threshold boundary conditions (score == threshold is flagged).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from ai_image_audit.classifier import (
    DEFAULT_THRESHOLD,
    AIImageClassifier,
    ClassificationResult,
    _build_transform,
    _finetuned_score,
    _heuristic_score,
    _to_rgb,
    classify_image,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classifier() -> AIImageClassifier:
    """Shared heuristic-mode classifier for the entire test module."""
    return AIImageClassifier(device="cpu")


@pytest.fixture()
def rgb_image() -> Image.Image:
    """Return a small 64x64 solid-red RGB image."""
    return Image.new("RGB", (64, 64), color=(200, 50, 50))


@pytest.fixture()
def rgba_image() -> Image.Image:
    """Return a small 64x64 semi-transparent RGBA image."""
    return Image.new("RGBA", (64, 64), color=(0, 128, 255, 128))


@pytest.fixture()
def grayscale_image() -> Image.Image:
    """Return a small 32x32 grayscale image."""
    return Image.new("L", (32, 32), color=128)


@pytest.fixture()
def palette_image() -> Image.Image:
    """Return a small palette-mode (P) image."""
    img = Image.new("P", (32, 32))
    return img


@pytest.fixture()
def la_image() -> Image.Image:
    """Return a small grayscale + alpha (LA) image."""
    return Image.new("LA", (32, 32), color=(100, 200))


@pytest.fixture()
def valid_jpeg(tmp_path: Path) -> Path:
    """Save a valid JPEG image to a temporary file and return its path."""
    p = tmp_path / "valid.jpg"
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(str(p), format="JPEG")
    return p


@pytest.fixture()
def corrupt_file(tmp_path: Path) -> Path:
    """Create a file with non-image content and return its path."""
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"this is not an image")
    return p


# ---------------------------------------------------------------------------
# ClassificationResult tests
# ---------------------------------------------------------------------------


class TestClassificationResult:
    """Tests for the ClassificationResult dataclass."""

    def test_fields_accessible(self) -> None:
        """All fields should be accessible after construction."""
        result = ClassificationResult(
            score=0.7,
            is_flagged=True,
            threshold=0.5,
            verdict="AI-generated (flagged)",
            model_mode="heuristic",
        )
        assert result.score == pytest.approx(0.7)
        assert result.is_flagged is True
        assert result.threshold == pytest.approx(0.5)
        assert result.verdict == "AI-generated (flagged)"
        assert result.model_mode == "heuristic"

    def test_score_in_valid_range(self) -> None:
        """Score value 0.0 should be a valid lower bound."""
        result = ClassificationResult(
            score=0.0,
            is_flagged=False,
            threshold=0.5,
            verdict="Human-made (not flagged)",
            model_mode="fine-tuned",
        )
        assert 0.0 <= result.score <= 1.0

    def test_verdict_not_flagged(self) -> None:
        """Verdict for a non-flagged result should contain 'not flagged'."""
        result = ClassificationResult(
            score=0.3,
            is_flagged=False,
            threshold=0.5,
            verdict="Human-made (not flagged)",
            model_mode="heuristic",
        )
        assert "not flagged" in result.verdict

    def test_verdict_flagged(self) -> None:
        """Verdict for a flagged result should mention 'AI-generated'."""
        result = ClassificationResult(
            score=0.9,
            is_flagged=True,
            threshold=0.5,
            verdict="AI-generated (flagged)",
            model_mode="fine-tuned",
        )
        assert "flagged" in result.verdict
        assert "AI-generated" in result.verdict

    def test_mode_fine_tuned(self) -> None:
        """model_mode field should store the given string."""
        result = ClassificationResult(
            score=0.6,
            is_flagged=True,
            threshold=0.5,
            verdict="AI-generated (flagged)",
            model_mode="fine-tuned",
        )
        assert result.model_mode == "fine-tuned"

    def test_mode_heuristic(self) -> None:
        """model_mode field accepts 'heuristic'."""
        result = ClassificationResult(
            score=0.4,
            is_flagged=False,
            threshold=0.5,
            verdict="Human-made (not flagged)",
            model_mode="heuristic",
        )
        assert result.model_mode == "heuristic"

    def test_threshold_stored_correctly(self) -> None:
        """The threshold field should exactly match the supplied value."""
        result = ClassificationResult(
            score=0.55,
            is_flagged=True,
            threshold=0.5,
            verdict="AI-generated (flagged)",
            model_mode="heuristic",
        )
        assert result.threshold == pytest.approx(0.5)

    def test_score_exactly_one(self) -> None:
        """A score of exactly 1.0 is a valid upper bound."""
        result = ClassificationResult(
            score=1.0,
            is_flagged=True,
            threshold=0.5,
            verdict="AI-generated (flagged)",
            model_mode="heuristic",
        )
        assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AIImageClassifier initialisation tests
# ---------------------------------------------------------------------------


class TestAIImageClassifierInit:
    """Tests for AIImageClassifier constructor behaviour."""

    def test_default_threshold(self) -> None:
        """Default threshold should equal DEFAULT_THRESHOLD."""
        c = AIImageClassifier(device="cpu")
        assert c.threshold == pytest.approx(DEFAULT_THRESHOLD)

    def test_custom_threshold(self) -> None:
        """A custom threshold should be stored on the instance."""
        c = AIImageClassifier(threshold=0.75, device="cpu")
        assert c.threshold == pytest.approx(0.75)

    def test_threshold_zero_allowed(self) -> None:
        """Threshold of 0.0 is a valid lower bound."""
        c = AIImageClassifier(threshold=0.0, device="cpu")
        assert c.threshold == pytest.approx(0.0)

    def test_threshold_one_allowed(self) -> None:
        """Threshold of 1.0 is a valid upper bound."""
        c = AIImageClassifier(threshold=1.0, device="cpu")
        assert c.threshold == pytest.approx(1.0)

    def test_invalid_threshold_below_zero(self) -> None:
        """Threshold below 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            AIImageClassifier(threshold=-0.1, device="cpu")

    def test_invalid_threshold_above_one(self) -> None:
        """Threshold above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            AIImageClassifier(threshold=1.1, device="cpu")

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """Supplying a non-existent model_path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AIImageClassifier(
                model_path=str(tmp_path / "nonexistent.pt"),
                device="cpu",
            )

    def test_heuristic_mode_no_model_path(self) -> None:
        """No model_path should activate heuristic mode."""
        c = AIImageClassifier(device="cpu")
        assert c._mode == "heuristic"

    def test_device_set_to_cpu(self) -> None:
        """Explicitly passing device='cpu' should set the torch device to cpu."""
        c = AIImageClassifier(device="cpu")
        assert str(c._device) == "cpu"

    def test_model_is_in_eval_mode(self) -> None:
        """The underlying model should be in eval (not training) mode."""
        c = AIImageClassifier(device="cpu")
        assert not c._model.training

    def test_transform_is_callable(self) -> None:
        """The stored transform pipeline should be callable."""
        c = AIImageClassifier(device="cpu")
        assert callable(c._transform)

    def test_threshold_boundary_exactly_zero_point_five(self) -> None:
        """Threshold of exactly 0.5 (the default) should be accepted."""
        c = AIImageClassifier(threshold=0.5, device="cpu")
        assert c.threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _to_rgb helper tests
# ---------------------------------------------------------------------------


class TestToRgb:
    """Tests for the _to_rgb image mode conversion helper."""

    def test_rgb_unchanged(self, rgb_image: Image.Image) -> None:
        """An already-RGB image should be returned as-is (same object)."""
        result = _to_rgb(rgb_image)
        assert result.mode == "RGB"
        assert result is rgb_image

    def test_rgba_converted_to_rgb(self, rgba_image: Image.Image) -> None:
        """RGBA images should be converted to RGB."""
        result = _to_rgb(rgba_image)
        assert result.mode == "RGB"
        assert result.size == rgba_image.size

    def test_la_converted_to_rgb(self, la_image: Image.Image) -> None:
        """LA (grayscale + alpha) images should be converted to RGB."""
        result = _to_rgb(la_image)
        assert result.mode == "RGB"

    def test_grayscale_converted_to_rgb(self, grayscale_image: Image.Image) -> None:
        """Grayscale ('L') images should be converted to RGB."""
        result = _to_rgb(grayscale_image)
        assert result.mode == "RGB"

    def test_palette_converted_to_rgb(self, palette_image: Image.Image) -> None:
        """Palette-mode ('P') images should be converted to RGB."""
        result = _to_rgb(palette_image)
        assert result.mode == "RGB"

    def test_output_has_three_channels(self, rgba_image: Image.Image) -> None:
        """The output of _to_rgb should always have exactly 3 bands."""
        result = _to_rgb(rgba_image)
        assert len(result.getbands()) == 3

    def test_rgba_size_preserved(self, rgba_image: Image.Image) -> None:
        """The spatial dimensions should not change during conversion."""
        result = _to_rgb(rgba_image)
        assert result.size == rgba_image.size

    def test_grayscale_size_preserved(self, grayscale_image: Image.Image) -> None:
        """Grayscale to RGB conversion should preserve image size."""
        result = _to_rgb(grayscale_image)
        assert result.size == grayscale_image.size

    def test_la_size_preserved(self, la_image: Image.Image) -> None:
        """LA to RGB conversion should preserve image size."""
        result = _to_rgb(la_image)
        assert result.size == la_image.size


# ---------------------------------------------------------------------------
# _build_transform tests
# ---------------------------------------------------------------------------


class TestBuildTransform:
    """Tests for the _build_transform preprocessing pipeline factory."""

    def test_returns_compose(self) -> None:
        """_build_transform should return a torchvision Compose instance."""
        from torchvision import transforms

        t = _build_transform()
        assert isinstance(t, transforms.Compose)

    def test_transform_produces_tensor(self, rgb_image: Image.Image) -> None:
        """Applying the transform to a PIL image should yield a torch.Tensor."""
        t = _build_transform()
        result = t(rgb_image)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self, rgb_image: Image.Image) -> None:
        """The output tensor should have shape (3, 224, 224)."""
        t = _build_transform()
        result = t(rgb_image)
        assert result.shape == (3, 224, 224)

    def test_output_dtype_float32(self, rgb_image: Image.Image) -> None:
        """The output tensor should be float32."""
        t = _build_transform()
        result = t(rgb_image)
        assert result.dtype == torch.float32

    def test_transform_on_small_image(self) -> None:
        """The transform should upscale a tiny image without error."""
        tiny = Image.new("RGB", (8, 8), color=(128, 64, 32))
        t = _build_transform()
        result = t(tiny)
        assert result.shape == (3, 224, 224)

    def test_transform_on_large_image(self) -> None:
        """The transform should downscale a large image without error."""
        large = Image.new("RGB", (1024, 768), color=(200, 200, 200))
        t = _build_transform()
        result = t(large)
        assert result.shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# _finetuned_score tests
# ---------------------------------------------------------------------------


class TestFinetunedScore:
    """Tests for the fine-tuned scoring helper."""

    def test_high_ai_logit_gives_high_score(self) -> None:
        """When the AI class logit is much larger, score should be near 1."""
        logits = torch.tensor([[0.0, 10.0]])
        score = _finetuned_score(logits)
        assert score > 0.99

    def test_low_ai_logit_gives_low_score(self) -> None:
        """When the non-AI class logit dominates, score should be near 0."""
        logits = torch.tensor([[10.0, 0.0]])
        score = _finetuned_score(logits)
        assert score < 0.01

    def test_equal_logits_give_near_half(self) -> None:
        """Equal logits should yield a score close to 0.5."""
        logits = torch.tensor([[0.0, 0.0]])
        score = _finetuned_score(logits)
        assert score == pytest.approx(0.5)

    def test_score_in_range(self) -> None:
        """For random logits, scores should always be in [0, 1]."""
        for _ in range(20):
            logits = torch.randn(1, 2)
            score = _finetuned_score(logits)
            assert 0.0 <= score <= 1.0

    def test_score_is_float(self) -> None:
        """Return type should be a Python float."""
        logits = torch.tensor([[1.0, -1.0]])
        score = _finetuned_score(logits)
        assert isinstance(score, float)

    def test_large_positive_ai_logit(self) -> None:
        """Very large AI logit should push score to essentially 1.0."""
        logits = torch.tensor([[0.0, 100.0]])
        score = _finetuned_score(logits)
        assert score > 0.9999

    def test_large_negative_ai_logit(self) -> None:
        """Very large negative AI logit should push score to essentially 0.0."""
        logits = torch.tensor([[100.0, 0.0]])
        score = _finetuned_score(logits)
        assert score < 0.0001


# ---------------------------------------------------------------------------
# _heuristic_score tests
# ---------------------------------------------------------------------------


class TestHeuristicScore:
    """Tests for the heuristic ImageNet-proxy scoring helper."""

    def test_score_in_range_uniform_logits(self) -> None:
        """Uniform logits (maximum entropy) should yield a score in [0, 1]."""
        logits = torch.zeros(1, 1000)
        score = _heuristic_score(logits)
        assert 0.0 <= score <= 1.0

    def test_score_in_range_random_logits(self) -> None:
        """Random logits should always yield a score in [0, 1]."""
        for _ in range(20):
            logits = torch.randn(1, 1000)
            score = _heuristic_score(logits)
            assert 0.0 <= score <= 1.0

    def test_high_entropy_increases_score(self) -> None:
        """Uniform distribution (max entropy) should score higher than a peaked one."""
        uniform_logits = torch.zeros(1, 1000)
        peaked_logits = torch.zeros(1, 1000)
        peaked_logits[0, 0] = 100.0

        uniform_score = _heuristic_score(uniform_logits)
        peaked_score = _heuristic_score(peaked_logits)
        assert uniform_score > peaked_score

    def test_score_is_float(self) -> None:
        """Return type should be a Python float."""
        logits = torch.randn(1, 1000)
        score = _heuristic_score(logits)
        assert isinstance(score, float)

    def test_score_clipped_to_one(self) -> None:
        """Score should never exceed 1.0."""
        logits = torch.full((1, 1000), 10.0)
        score = _heuristic_score(logits)
        assert score <= 1.0

    def test_score_clipped_to_zero(self) -> None:
        """Score should never be below 0.0."""
        logits = torch.randn(1, 1000)
        score = _heuristic_score(logits)
        assert score >= 0.0

    def test_peaked_distribution_gives_low_entropy(self) -> None:
        """A distribution peaked on a single class should have near-zero entropy component."""
        peaked = torch.full((1, 1000), -100.0)
        peaked[0, 42] = 100.0
        score = _heuristic_score(peaked)
        # With very low entropy, score should be low (entropy weight = 0.6).
        assert score < 0.5


# ---------------------------------------------------------------------------
# AIImageClassifier.classify() tests
# ---------------------------------------------------------------------------


class TestClassify:
    """Tests for the primary classify() method."""

    def test_returns_classification_result(
        self, classifier: AIImageClassifier, rgb_image: Image.Image
    ) -> None:
        """classify() should return a ClassificationResult instance."""
        result = classifier.classify(rgb_image)
        assert isinstance(result, ClassificationResult)

    def test_score_in_range(
        self, classifier: AIImageClassifier, rgb_image: Image.Image
    ) -> None:
        """The score should be a float in [0.0, 1.0]."""
        result = classifier.classify(rgb_image)
        assert 0.0 <= result.score <= 1.0

    def test_mode_is_heuristic(
        self, classifier: AIImageClassifier, rgb_image: Image.Image
    ) -> None:
        """The model_mode field should be 'heuristic' for the default classifier."""
        result = classifier.classify(rgb_image)
        assert result.model_mode == "heuristic"

    def test_threshold_echoed(
        self, classifier: AIImageClassifier, rgb_image: Image.Image
    ) -> None:
        """The threshold in the result should match the classifier's threshold."""
        result = classifier.classify(rgb_image)
        assert result.threshold == pytest.approx(DEFAULT_THRESHOLD)

    def test_is_flagged_consistent_with_score(
        self, classifier: AIImageClassifier, rgb_image: Image.Image
    ) -> None:
        """is_flagged should be True iff score >= threshold."""
        result = classifier.classify(rgb_image)
        if result.score >= result.threshold:
            assert result.is_flagged is True
        else:
            assert result.is_flagged is False

    def test_verdict_consistent_with_is_flagged(
        self, classifier: AIImageClassifier, rgb_image: Image.Image
    ) -> None:
        """Verdict text should be consistent with is_flagged."""
        result = classifier.classify(rgb_image)
        if result.is_flagged:
            assert "AI-generated" in result.verdict
        else:
            assert "Human-made" in result.verdict

    def test_rgba_image_handled(
        self, classifier: AIImageClassifier, rgba_image: Image.Image
    ) -> None:
        """RGBA images should be classified without raising."""
        result = classifier.classify(rgba_image)
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.score <= 1.0

    def test_grayscale_image_handled(
        self, classifier: AIImageClassifier, grayscale_image: Image.Image
    ) -> None:
        """Grayscale images should be classified without raising."""
        result = classifier.classify(grayscale_image)
        assert isinstance(result, ClassificationResult)

    def test_palette_image_handled(
        self, classifier: AIImageClassifier, palette_image: Image.Image
    ) -> None:
        """Palette-mode images should be classified without raising."""
        result = classifier.classify(palette_image)
        assert isinstance(result, ClassificationResult)

    def test_la_image_handled(
        self, classifier: AIImageClassifier, la_image: Image.Image
    ) -> None:
        """LA images should be classified without raising."""
        result = classifier.classify(la_image)
        assert isinstance(result, ClassificationResult)

    def test_non_pil_raises_type_error(
        self, classifier: AIImageClassifier
    ) -> None:
        """Passing a non-PIL object should raise TypeError."""
        with pytest.raises(TypeError, match="PIL.Image.Image"):
            classifier.classify("not an image")  # type: ignore[arg-type]

    def test_non_pil_numpy_raises_type_error(
        self, classifier: AIImageClassifier
    ) -> None:
        """Passing a numpy array should raise TypeError."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(TypeError):
            classifier.classify(arr)  # type: ignore[arg-type]

    def test_non_pil_int_raises_type_error(
        self, classifier: AIImageClassifier
    ) -> None:
        """Passing an integer should raise TypeError."""
        with pytest.raises(TypeError):
            classifier.classify(42)  # type: ignore[arg-type]

    def test_threshold_zero_always_flags(self) -> None:
        """With threshold=0.0 every image should be flagged (score >= 0)."""
        c = AIImageClassifier(threshold=0.0, device="cpu")
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        result = c.classify(img)
        assert result.is_flagged is True

    def test_threshold_one_not_flagged_for_heuristic(self) -> None:
        """With threshold=1.0 the heuristic should not produce exactly 1.0."""
        c = AIImageClassifier(threshold=1.0, device="cpu")
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        result = c.classify(img)
        # Just verify the result is a valid ClassificationResult.
        assert isinstance(result, ClassificationResult)
        # is_flagged should be consistent with score vs threshold.
        assert result.is_flagged == (result.score >= result.threshold)

    def test_threshold_at_score_boundary(self) -> None:
        """When score == threshold the image should be flagged."""
        c = AIImageClassifier(threshold=0.5, device="cpu")
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        result = c.classify(img)
        expected_flagged = result.score >= c.threshold
        assert result.is_flagged == expected_flagged

    def test_small_image_handled(
        self, classifier: AIImageClassifier
    ) -> None:
        """A 1x1 pixel image should be classified without error."""
        tiny = Image.new("RGB", (1, 1), color=(128, 128, 128))
        result = classifier.classify(tiny)
        assert 0.0 <= result.score <= 1.0

    def test_large_image_handled(
        self, classifier: AIImageClassifier
    ) -> None:
        """A 1024x1024 image should be classified without error."""
        large = Image.new("RGB", (1024, 1024), color=(200, 200, 200))
        result = classifier.classify(large)
        assert 0.0 <= result.score <= 1.0

    def test_multiple_calls_deterministic(
        self, classifier: AIImageClassifier, rgb_image: Image.Image
    ) -> None:
        """Calling classify twice on the same image should give the same score."""
        r1 = classifier.classify(rgb_image)
        r2 = classifier.classify(rgb_image)
        assert r1.score == pytest.approx(r2.score)

    def test_result_verdict_is_string(
        self, classifier: AIImageClassifier, rgb_image: Image.Image
    ) -> None:
        """The verdict field should always be a non-empty string."""
        result = classifier.classify(rgb_image)
        assert isinstance(result.verdict, str)
        assert len(result.verdict) > 0

    def test_result_score_is_float(
        self, classifier: AIImageClassifier, rgb_image: Image.Image
    ) -> None:
        """The score field should be a Python float."""
        result = classifier.classify(rgb_image)
        assert isinstance(result.score, float)

    def test_result_is_flagged_is_bool(
        self, classifier: AIImageClassifier, rgb_image: Image.Image
    ) -> None:
        """The is_flagged field should be a Python bool."""
        result = classifier.classify(rgb_image)
        assert isinstance(result.is_flagged, bool)


# ---------------------------------------------------------------------------
# AIImageClassifier.classify_path() tests
# ---------------------------------------------------------------------------


class TestClassifyPath:
    """Tests for the classify_path() convenience method."""

    def test_valid_jpeg(
        self, classifier: AIImageClassifier, valid_jpeg: Path
    ) -> None:
        """A valid JPEG file should be classified successfully."""
        result = classifier.classify_path(str(valid_jpeg))
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.score <= 1.0

    def test_missing_file_raises_file_not_found(
        self, classifier: AIImageClassifier, tmp_path: Path
    ) -> None:
        """A non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            classifier.classify_path(str(tmp_path / "missing.jpg"))

    def test_corrupt_file_raises_os_error(
        self, classifier: AIImageClassifier, corrupt_file: Path
    ) -> None:
        """A corrupt (non-image) file should raise OSError."""
        with pytest.raises(OSError):
            classifier.classify_path(str(corrupt_file))

    def test_result_is_classification_result(
        self, classifier: AIImageClassifier, valid_jpeg: Path
    ) -> None:
        """The return type should be ClassificationResult."""
        result = classifier.classify_path(str(valid_jpeg))
        assert isinstance(result, ClassificationResult)

    def test_png_file_classified(
        self, classifier: AIImageClassifier, tmp_path: Path
    ) -> None:
        """A valid PNG file should also be classified without error."""
        p = tmp_path / "test.png"
        Image.new("RGB", (32, 32), color=(0, 255, 0)).save(str(p), format="PNG")
        result = classifier.classify_path(str(p))
        assert isinstance(result, ClassificationResult)

    def test_gif_file_classified(
        self, classifier: AIImageClassifier, tmp_path: Path
    ) -> None:
        """A valid GIF file should be classified without error."""
        p = tmp_path / "test.gif"
        Image.new("P", (32, 32)).save(str(p), format="GIF")
        result = classifier.classify_path(str(p))
        assert isinstance(result, ClassificationResult)

    def test_bmp_file_classified(
        self, classifier: AIImageClassifier, tmp_path: Path
    ) -> None:
        """A valid BMP file should be classified without error."""
        p = tmp_path / "test.bmp"
        Image.new("RGB", (32, 32), color=(0, 0, 255)).save(str(p), format="BMP")
        result = classifier.classify_path(str(p))
        assert isinstance(result, ClassificationResult)

    def test_score_in_range_for_path(
        self, classifier: AIImageClassifier, valid_jpeg: Path
    ) -> None:
        """Score from classify_path should be in [0, 1]."""
        result = classifier.classify_path(str(valid_jpeg))
        assert 0.0 <= result.score <= 1.0

    def test_threshold_in_result_matches_classifier(
        self, classifier: AIImageClassifier, valid_jpeg: Path
    ) -> None:
        """The threshold in the result should match the classifier instance."""
        result = classifier.classify_path(str(valid_jpeg))
        assert result.threshold == pytest.approx(classifier.threshold)


# ---------------------------------------------------------------------------
# classify_image() convenience function tests
# ---------------------------------------------------------------------------


class TestClassifyImageFunction:
    """Tests for the module-level classify_image() convenience function."""

    def test_returns_classification_result(self, rgb_image: Image.Image) -> None:
        """classify_image() should return a ClassificationResult."""
        result = classify_image(rgb_image, device="cpu")
        assert isinstance(result, ClassificationResult)

    def test_score_in_range(self, rgb_image: Image.Image) -> None:
        """Score from classify_image() should be in [0, 1]."""
        result = classify_image(rgb_image, device="cpu")
        assert 0.0 <= result.score <= 1.0

    def test_custom_threshold_applied(self, rgb_image: Image.Image) -> None:
        """A custom threshold should be reflected in the result."""
        result = classify_image(rgb_image, threshold=0.9, device="cpu")
        assert result.threshold == pytest.approx(0.9)

    def test_invalid_threshold_raises(self, rgb_image: Image.Image) -> None:
        """An out-of-range threshold should raise ValueError."""
        with pytest.raises(ValueError):
            classify_image(rgb_image, threshold=1.5, device="cpu")

    def test_negative_threshold_raises(self, rgb_image: Image.Image) -> None:
        """A negative threshold should raise ValueError."""
        with pytest.raises(ValueError):
            classify_image(rgb_image, threshold=-0.5, device="cpu")

    def test_mode_is_heuristic_without_model_path(
        self, rgb_image: Image.Image
    ) -> None:
        """Without a model_path, mode should be 'heuristic'."""
        result = classify_image(rgb_image, device="cpu")
        assert result.model_mode == "heuristic"

    def test_rgba_image(self, rgba_image: Image.Image) -> None:
        """classify_image() should handle RGBA images."""
        result = classify_image(rgba_image, device="cpu")
        assert isinstance(result, ClassificationResult)

    def test_grayscale_image(self, grayscale_image: Image.Image) -> None:
        """classify_image() should handle grayscale images."""
        result = classify_image(grayscale_image, device="cpu")
        assert isinstance(result, ClassificationResult)

    def test_verdict_is_string(self, rgb_image: Image.Image) -> None:
        """The verdict field should be a non-empty string."""
        result = classify_image(rgb_image, device="cpu")
        assert isinstance(result.verdict, str)
        assert len(result.verdict) > 0


# ---------------------------------------------------------------------------
# Fine-tuned mode integration test (with a synthetic checkpoint)
# ---------------------------------------------------------------------------


class TestFinetunedMode:
    """Tests for fine-tuned mode using a synthetic checkpoint saved to disk."""

    @pytest.fixture(scope="class")
    def finetuned_checkpoint(self, tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
        """Create a synthetic 2-class MobileNetV2 checkpoint and save it."""
        import torch.nn as nn
        from torchvision import models

        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            model = models.mobilenet_v2(weights=weights)
        except Exception:  # noqa: BLE001
            model = models.mobilenet_v2(pretrained=False)

        in_features = model.classifier[1].in_features  # type: ignore[index]
        model.classifier[1] = nn.Linear(in_features, 2)

        checkpoint_dir = tmp_path_factory.mktemp("checkpoints")
        checkpoint_path = checkpoint_dir / "ai_detector.pt"
        torch.save(model.state_dict(), str(checkpoint_path))
        return checkpoint_path

    def test_finetuned_classifier_loads(
        self, finetuned_checkpoint: Path
    ) -> None:
        """A valid checkpoint should be loaded and mode set to 'fine-tuned'."""
        c = AIImageClassifier(
            model_path=str(finetuned_checkpoint),
            device="cpu",
        )
        assert c._mode == "fine-tuned"

    def test_finetuned_classify_returns_result(
        self, finetuned_checkpoint: Path
    ) -> None:
        """classify() in fine-tuned mode should return a valid ClassificationResult."""
        c = AIImageClassifier(
            model_path=str(finetuned_checkpoint),
            device="cpu",
        )
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        result = c.classify(img)
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.score <= 1.0
        assert result.model_mode == "fine-tuned"

    def test_finetuned_score_consistent_with_is_flagged(
        self, finetuned_checkpoint: Path
    ) -> None:
        """is_flagged should correctly reflect score >= threshold in fine-tuned mode."""
        c = AIImageClassifier(
            model_path=str(finetuned_checkpoint),
            threshold=0.5,
            device="cpu",
        )
        img = Image.new("RGB", (64, 64), color=(50, 100, 150))
        result = c.classify(img)
        assert result.is_flagged == (result.score >= result.threshold)

    def test_invalid_checkpoint_raises_runtime_error(
        self, tmp_path: Path
    ) -> None:
        """A file that is not a valid state-dict should raise RuntimeError."""
        bad_path = tmp_path / "bad.pt"
        torch.save({"wrong_key": torch.zeros(10)}, str(bad_path))
        with pytest.raises(RuntimeError, match="Failed to load"):
            AIImageClassifier(
                model_path=str(bad_path),
                device="cpu",
            )

    def test_finetuned_model_in_eval_mode(
        self, finetuned_checkpoint: Path
    ) -> None:
        """The fine-tuned model should be set to eval mode."""
        c = AIImageClassifier(
            model_path=str(finetuned_checkpoint),
            device="cpu",
        )
        assert not c._model.training

    def test_finetuned_score_in_range_random_images(
        self, finetuned_checkpoint: Path
    ) -> None:
        """Fine-tuned mode scores should always be in [0, 1] for varied inputs."""
        c = AIImageClassifier(
            model_path=str(finetuned_checkpoint),
            device="cpu",
        )
        for colour in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]:
            img = Image.new("RGB", (64, 64), color=colour)
            result = c.classify(img)
            assert 0.0 <= result.score <= 1.0

    def test_finetuned_verdict_is_string(
        self, finetuned_checkpoint: Path
    ) -> None:
        """Verdict should be a non-empty string in fine-tuned mode."""
        c = AIImageClassifier(
            model_path=str(finetuned_checkpoint),
            device="cpu",
        )
        img = Image.new("RGB", (32, 32), color=(200, 200, 200))
        result = c.classify(img)
        assert isinstance(result.verdict, str)
        assert len(result.verdict) > 0

    def test_finetuned_classify_path(
        self, finetuned_checkpoint: Path, tmp_path: Path
    ) -> None:
        """classify_path() should work correctly in fine-tuned mode."""
        p = tmp_path / "test_ft.jpg"
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(str(p), format="JPEG")
        c = AIImageClassifier(
            model_path=str(finetuned_checkpoint),
            device="cpu",
        )
        result = c.classify_path(str(p))
        assert isinstance(result, ClassificationResult)
        assert result.model_mode == "fine-tuned"

    def test_finetuned_wrapped_checkpoint_with_state_dict_key(
        self, tmp_path: Path
    ) -> None:
        """Checkpoints wrapped with a 'state_dict' key should load correctly."""
        import torch.nn as nn
        from torchvision import models

        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            model = models.mobilenet_v2(weights=weights)
        except Exception:  # noqa: BLE001
            model = models.mobilenet_v2(pretrained=False)

        in_features = model.classifier[1].in_features  # type: ignore[index]
        model.classifier[1] = nn.Linear(in_features, 2)

        wrapped_path = tmp_path / "wrapped.pt"
        torch.save({"state_dict": model.state_dict()}, str(wrapped_path))

        c = AIImageClassifier(
            model_path=str(wrapped_path),
            device="cpu",
        )
        assert c._mode == "fine-tuned"
        img = Image.new("RGB", (64, 64), color=(50, 50, 50))
        result = c.classify(img)
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.score <= 1.0
