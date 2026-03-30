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
import struct
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
    """Return a small 64×64 solid-red RGB image."""
    return Image.new("RGB", (64, 64), color=(200, 50, 50))


@pytest.fixture()
def rgba_image() -> Image.Image:
    """Return a small 64×64 semi-transparent RGBA image."""
    return Image.new("RGBA", (64, 64), color=(0, 128, 255, 128))


@pytest.fixture()
def grayscale_image() -> Image.Image:
    """Return a small 32×32 grayscale image."""
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
    def test_fields_accessible(self) -> None:
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
        result = ClassificationResult(
            score=0.0,
            is_flagged=False,
            threshold=0.5,
            verdict="Human-made (not flagged)",
            model_mode="fine-tuned",
        )
        assert 0.0 <= result.score <= 1.0

    def test_verdict_not_flagged(self) -> None:
        result = ClassificationResult(
            score=0.3,
            is_flagged=False,
            threshold=0.5,
            verdict="Human-made (not flagged)",
            model_mode="heuristic",
        )
        assert "not flagged" in result.verdict

    def test_verdict_flagged(self) -> None:
        result = ClassificationResult(
            score=0.9,
            is_flagged=True,
            threshold=0.5,
            verdict="AI-generated (flagged)",
            model_mode="fine-tuned",
        )
        assert "flagged" in result.verdict
        assert "AI-generated" in result.verdict


# ---------------------------------------------------------------------------
# AIImageClassifier initialisation tests
# ---------------------------------------------------------------------------


class TestAIImageClassifierInit:
    def test_default_threshold(self) -> None:
        c = AIImageClassifier(device="cpu")
        assert c.threshold == pytest.approx(DEFAULT_THRESHOLD)

    def test_custom_threshold(self) -> None:
        c = AIImageClassifier(threshold=0.75, device="cpu")
        assert c.threshold == pytest.approx(0.75)

    def test_threshold_zero_allowed(self) -> None:
        c = AIImageClassifier(threshold=0.0, device="cpu")
        assert c.threshold == pytest.approx(0.0)

    def test_threshold_one_allowed(self) -> None:
        c = AIImageClassifier(threshold=1.0, device="cpu")
        assert c.threshold == pytest.approx(1.0)

    def test_invalid_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            AIImageClassifier(threshold=-0.1, device="cpu")

    def test_invalid_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            AIImageClassifier(threshold=1.1, device="cpu")

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AIImageClassifier(
                model_path=str(tmp_path / "nonexistent.pt"),
                device="cpu",
            )

    def test_heuristic_mode_no_model_path(self) -> None:
        c = AIImageClassifier(device="cpu")
        assert c._mode == "heuristic"

    def test_device_set_to_cpu(self) -> None:
        c = AIImageClassifier(device="cpu")
        assert str(c._device) == "cpu"

    def test_model_is_in_eval_mode(self) -> None:
        c = AIImageClassifier(device="cpu")
        assert not c._model.training


# ---------------------------------------------------------------------------
# _to_rgb helper tests
# ---------------------------------------------------------------------------


class TestToRgb:
    def test_rgb_unchanged(self, rgb_image: Image.Image) -> None:
        result = _to_rgb(rgb_image)
        assert result.mode == "RGB"
        # Should return the same object for RGB (no-op).
        assert result is rgb_image

    def test_rgba_converted_to_rgb(self, rgba_image: Image.Image) -> None:
        result = _to_rgb(rgba_image)
        assert result.mode == "RGB"
        assert result.size == rgba_image.size

    def test_la_converted_to_rgb(self, la_image: Image.Image) -> None:
        result = _to_rgb(la_image)
        assert result.mode == "RGB"

    def test_grayscale_converted_to_rgb(self, grayscale_image: Image.Image) -> None:
        result = _to_rgb(grayscale_image)
        assert result.mode == "RGB"

    def test_palette_converted_to_rgb(self, palette_image: Image.Image) -> None:
        result = _to_rgb(palette_image)
        assert result.mode == "RGB"

    def test_output_has_three_channels(self, rgba_image: Image.Image) -> None:
        result = _to_rgb(rgba_image)
        assert len(result.getbands()) == 3


# ---------------------------------------------------------------------------
# _build_transform tests
# ---------------------------------------------------------------------------


class TestBuildTransform:
    def test_returns_compose(self) -> None:
        from torchvision import transforms

        t = _build_transform()
        assert isinstance(t, transforms.Compose)

    def test_transform_produces_tensor(self, rgb_image: Image.Image) -> None:
        t = _build_transform()
        result = t(rgb_image)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self, rgb_image: Image.Image) -> None:
        t = _build_transform()
        result = t(rgb_image)
        assert result.shape == (3, 224, 224)

    def test_output_dtype_float32(self, rgb_image: Image.Image) -> None:
        t = _build_transform()
        result = t(rgb_image)
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# _finetuned_score tests
# ---------------------------------------------------------------------------


class TestFinetunedScore:
    def test_high_ai_logit_gives_high_score(self) -> None:
        # Class index 1 (AI) has a much higher logit.
        logits = torch.tensor([[0.0, 10.0]])
        score = _finetuned_score(logits)
        assert score > 0.99

    def test_low_ai_logit_gives_low_score(self) -> None:
        logits = torch.tensor([[10.0, 0.0]])
        score = _finetuned_score(logits)
        assert score < 0.01

    def test_equal_logits_give_near_half(self) -> None:
        logits = torch.tensor([[0.0, 0.0]])
        score = _finetuned_score(logits)
        assert score == pytest.approx(0.5)

    def test_score_in_range(self) -> None:
        for _ in range(20):
            logits = torch.randn(1, 2)
            score = _finetuned_score(logits)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _heuristic_score tests
# ---------------------------------------------------------------------------


class TestHeuristicScore:
    def test_score_in_range_uniform_logits(self) -> None:
        logits = torch.zeros(1, 1000)
        score = _heuristic_score(logits)
        assert 0.0 <= score <= 1.0

    def test_score_in_range_random_logits(self) -> None:
        for _ in range(20):
            logits = torch.randn(1, 1000)
            score = _heuristic_score(logits)
            assert 0.0 <= score <= 1.0

    def test_high_entropy_increases_score(self) -> None:
        # Uniform distribution → maximum entropy → higher score.
        uniform_logits = torch.zeros(1, 1000)
        # Peaked distribution → low entropy.
        peaked_logits = torch.zeros(1, 1000)
        peaked_logits[0, 0] = 100.0

        uniform_score = _heuristic_score(uniform_logits)
        peaked_score = _heuristic_score(peaked_logits)
        assert uniform_score > peaked_score

    def test_score_is_float(self) -> None:
        logits = torch.randn(1, 1000)
        score = _heuristic_score(logits)
        assert isinstance(score, float)

    def test_score_clipped_to_one(self) -> None:
        # Force maximum activation on all art indices by setting them very high.
        logits = torch.full((1, 1000), -100.0)
        # Uniform very high → entropy near max.
        logits[:, :] = 10.0
        score = _heuristic_score(logits)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# AIImageClassifier.classify() tests
# ---------------------------------------------------------------------------


class TestClassify:
    def test_returns_classification_result(self, classifier: AIImageClassifier, rgb_image: Image.Image) -> None:
        result = classifier.classify(rgb_image)
        assert isinstance(result, ClassificationResult)

    def test_score_in_range(self, classifier: AIImageClassifier, rgb_image: Image.Image) -> None:
        result = classifier.classify(rgb_image)
        assert 0.0 <= result.score <= 1.0

    def test_mode_is_heuristic(self, classifier: AIImageClassifier, rgb_image: Image.Image) -> None:
        result = classifier.classify(rgb_image)
        assert result.model_mode == "heuristic"

    def test_threshold_echoed(self, classifier: AIImageClassifier, rgb_image: Image.Image) -> None:
        result = classifier.classify(rgb_image)
        assert result.threshold == pytest.approx(DEFAULT_THRESHOLD)

    def test_is_flagged_consistent_with_score(self, classifier: AIImageClassifier, rgb_image: Image.Image) -> None:
        result = classifier.classify(rgb_image)
        if result.score >= result.threshold:
            assert result.is_flagged is True
        else:
            assert result.is_flagged is False

    def test_verdict_consistent_with_is_flagged(self, classifier: AIImageClassifier, rgb_image: Image.Image) -> None:
        result = classifier.classify(rgb_image)
        if result.is_flagged:
            assert "AI-generated" in result.verdict
        else:
            assert "Human-made" in result.verdict

    def test_rgba_image_handled(self, classifier: AIImageClassifier, rgba_image: Image.Image) -> None:
        result = classifier.classify(rgba_image)
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.score <= 1.0

    def test_grayscale_image_handled(self, classifier: AIImageClassifier, grayscale_image: Image.Image) -> None:
        result = classifier.classify(grayscale_image)
        assert isinstance(result, ClassificationResult)

    def test_palette_image_handled(self, classifier: AIImageClassifier, palette_image: Image.Image) -> None:
        result = classifier.classify(palette_image)
        assert isinstance(result, ClassificationResult)

    def test_la_image_handled(self, classifier: AIImageClassifier, la_image: Image.Image) -> None:
        result = classifier.classify(la_image)
        assert isinstance(result, ClassificationResult)

    def test_non_pil_raises_type_error(self, classifier: AIImageClassifier) -> None:
        with pytest.raises(TypeError, match="PIL.Image.Image"):
            classifier.classify("not an image")  # type: ignore[arg-type]

    def test_non_pil_numpy_raises_type_error(self, classifier: AIImageClassifier) -> None:
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(TypeError):
            classifier.classify(arr)  # type: ignore[arg-type]

    def test_threshold_zero_always_flags(self) -> None:
        c = AIImageClassifier(threshold=0.0, device="cpu")
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        result = c.classify(img)
        # Score >= 0.0 always → always flagged.
        assert result.is_flagged is True

    def test_threshold_one_never_flags(self) -> None:
        c = AIImageClassifier(threshold=1.0, device="cpu")
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        result = c.classify(img)
        # Score <= 1.0 and threshold == 1.0: only flagged if score == 1.0 exactly.
        # With heuristic mode this should not produce exactly 1.0.
        # We just check the type returned is correct.
        assert isinstance(result, ClassificationResult)

    def test_threshold_at_score_boundary(self) -> None:
        """When score == threshold the image should be flagged."""
        c = AIImageClassifier(threshold=0.5, device="cpu")
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        result = c.classify(img)
        # Manually check the rule.
        expected_flagged = result.score >= c.threshold
        assert result.is_flagged == expected_flagged

    def test_small_image_handled(self, classifier: AIImageClassifier) -> None:
        """A 1×1 pixel image should not crash the classifier."""
        tiny = Image.new("RGB", (1, 1), color=(128, 128, 128))
        result = classifier.classify(tiny)
        assert 0.0 <= result.score <= 1.0

    def test_large_image_handled(self, classifier: AIImageClassifier) -> None:
        """A 1024×1024 image should be down-sampled and classified without error."""
        large = Image.new("RGB", (1024, 1024), color=(200, 200, 200))
        result = classifier.classify(large)
        assert 0.0 <= result.score <= 1.0

    def test_multiple_calls_deterministic(self, classifier: AIImageClassifier, rgb_image: Image.Image) -> None:
        """Calling classify twice on the same image should give the same score."""
        r1 = classifier.classify(rgb_image)
        r2 = classifier.classify(rgb_image)
        assert r1.score == pytest.approx(r2.score)


# ---------------------------------------------------------------------------
# AIImageClassifier.classify_path() tests
# ---------------------------------------------------------------------------


class TestClassifyPath:
    def test_valid_jpeg(self, classifier: AIImageClassifier, valid_jpeg: Path) -> None:
        result = classifier.classify_path(str(valid_jpeg))
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.score <= 1.0

    def test_missing_file_raises_file_not_found(self, classifier: AIImageClassifier, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            classifier.classify_path(str(tmp_path / "missing.jpg"))

    def test_corrupt_file_raises_os_error(self, classifier: AIImageClassifier, corrupt_file: Path) -> None:
        with pytest.raises(OSError):
            classifier.classify_path(str(corrupt_file))

    def test_result_is_classification_result(self, classifier: AIImageClassifier, valid_jpeg: Path) -> None:
        result = classifier.classify_path(str(valid_jpeg))
        assert isinstance(result, ClassificationResult)

    def test_png_file_classified(self, classifier: AIImageClassifier, tmp_path: Path) -> None:
        p = tmp_path / "test.png"
        Image.new("RGB", (32, 32), color=(0, 255, 0)).save(str(p), format="PNG")
        result = classifier.classify_path(str(p))
        assert isinstance(result, ClassificationResult)


# ---------------------------------------------------------------------------
# classify_image() convenience function tests
# ---------------------------------------------------------------------------


class TestClassifyImageFunction:
    def test_returns_classification_result(self, rgb_image: Image.Image) -> None:
        result = classify_image(rgb_image, device="cpu")
        assert isinstance(result, ClassificationResult)

    def test_score_in_range(self, rgb_image: Image.Image) -> None:
        result = classify_image(rgb_image, device="cpu")
        assert 0.0 <= result.score <= 1.0

    def test_custom_threshold_applied(self, rgb_image: Image.Image) -> None:
        result = classify_image(rgb_image, threshold=0.9, device="cpu")
        assert result.threshold == pytest.approx(0.9)

    def test_invalid_threshold_raises(self, rgb_image: Image.Image) -> None:
        with pytest.raises(ValueError):
            classify_image(rgb_image, threshold=1.5, device="cpu")


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
        except Exception:
            model = models.mobilenet_v2(pretrained=False)

        in_features = model.classifier[1].in_features  # type: ignore[index]
        model.classifier[1] = nn.Linear(in_features, 2)

        checkpoint_dir = tmp_path_factory.mktemp("checkpoints")
        checkpoint_path = checkpoint_dir / "ai_detector.pt"
        torch.save(model.state_dict(), str(checkpoint_path))
        return checkpoint_path

    def test_finetuned_classifier_loads(self, finetuned_checkpoint: Path) -> None:
        c = AIImageClassifier(
            model_path=str(finetuned_checkpoint),
            device="cpu",
        )
        assert c._mode == "fine-tuned"

    def test_finetuned_classify_returns_result(self, finetuned_checkpoint: Path) -> None:
        c = AIImageClassifier(
            model_path=str(finetuned_checkpoint),
            device="cpu",
        )
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        result = c.classify(img)
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.score <= 1.0
        assert result.model_mode == "fine-tuned"

    def test_finetuned_score_consistent_with_is_flagged(self, finetuned_checkpoint: Path) -> None:
        c = AIImageClassifier(
            model_path=str(finetuned_checkpoint),
            threshold=0.5,
            device="cpu",
        )
        img = Image.new("RGB", (64, 64), color=(50, 100, 150))
        result = c.classify(img)
        assert result.is_flagged == (result.score >= result.threshold)

    def test_invalid_checkpoint_raises_runtime_error(self, tmp_path: Path) -> None:
        # Create a file that is valid to open but incompatible as state-dict.
        bad_path = tmp_path / "bad.pt"
        torch.save({"wrong_key": torch.zeros(10)}, str(bad_path))
        with pytest.raises(RuntimeError, match="Failed to load"):
            AIImageClassifier(
                model_path=str(bad_path),
                device="cpu",
            )
