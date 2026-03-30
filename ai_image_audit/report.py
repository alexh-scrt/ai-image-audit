"""Per-image metadata report generation for the AI Image Audit tool.

This module provides functions and data structures for generating rich
metadata reports for image assets.  Each report captures:

- File-level metadata: path/URL, file size, format, mode.
- Visual metadata: dimensions (width × height), aspect ratio.
- Colour statistics: dominant palette (up to N colours via quantisation),
  mean and standard deviation per channel (R, G, B).
- AI classification result: score, verdict, threshold, model mode.

The primary entry point is :func:`generate_report`, which accepts a
:class:`~ai_image_audit.scanner.ImageRef` and a
:class:`~ai_image_audit.classifier.ClassificationResult` and returns a
:class:`ImageReport` dataclass.

Typical usage::

    from PIL import Image
    from ai_image_audit.scanner import ImageRef
    from ai_image_audit.classifier import AIImageClassifier
    from ai_image_audit.report import generate_report

    ref = ImageRef(source="/assets/artwork.jpg", is_remote=False, origin="/assets")
    classifier = AIImageClassifier(device="cpu")

    with Image.open(ref.source) as img:
        result = classifier.classify(img)
        report = generate_report(ref, result, img)

    print(report.to_dict())
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ai_image_audit.classifier import ClassificationResult
from ai_image_audit.scanner import ImageRef

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default number of dominant colours to extract from each image.
DEFAULT_PALETTE_COLOURS: int = 5

#: Number of colours used internally during palette quantisation.
_QUANTISE_COLOURS: int = 16


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ColourStats:
    """Per-channel colour statistics for an image.

    All values are computed on the RGB representation of the image.

    Attributes:
        mean_r: Mean pixel value for the red channel (0–255).
        mean_g: Mean pixel value for the green channel (0–255).
        mean_b: Mean pixel value for the blue channel (0–255).
        std_r: Standard deviation of the red channel (0–255).
        std_g: Standard deviation of the green channel (0–255).
        std_b: Standard deviation of the blue channel (0–255).
        dominant_colours: List of up to :data:`DEFAULT_PALETTE_COLOURS`
            dominant colours, each represented as an ``(R, G, B)`` tuple
            of integers in ``[0, 255]``.
    """

    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float
    dominant_colours: List[Tuple[int, int, int]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            A dictionary with all colour statistics.
        """
        return {
            "mean_r": round(self.mean_r, 2),
            "mean_g": round(self.mean_g, 2),
            "mean_b": round(self.mean_b, 2),
            "std_r": round(self.std_r, 2),
            "std_g": round(self.std_g, 2),
            "std_b": round(self.std_b, 2),
            "dominant_colours": [
                list(colour) for colour in self.dominant_colours
            ],
        }


@dataclass
class ImageReport:
    """Full metadata report for a single image asset.

    Attributes:
        source: The original source string (filesystem path or URL) from
            :class:`~ai_image_audit.scanner.ImageRef`.
        is_remote: Whether the image was sourced from a remote URL.
        origin: The root directory or base URL the image was discovered from.
        alt_text: The ``alt`` attribute from the originating ``<img>`` tag,
            if available.
        extension: Lowercase file extension including the leading dot.
        width: Image width in pixels.
        height: Image height in pixels.
        aspect_ratio: Width divided by height, rounded to 4 decimal places.
            ``None`` if height is zero (degenerate image).
        file_size_bytes: File size in bytes.  For remote images this is the
            size of the downloaded content; ``None`` if unavailable.
        format: PIL format string (e.g. ``"JPEG"``, ``"PNG"``), or
            ``"UNKNOWN"`` if PIL cannot determine it.
        mode: PIL image mode string (e.g. ``"RGB"``, ``"RGBA"``, ``"L"``).
        colour_stats: :class:`ColourStats` instance with per-channel
            statistics and dominant colour palette.
        ai_score: The raw AI-likelihood score from the classifier (0–1).
        ai_is_flagged: Whether the image was flagged as likely AI-generated.
        ai_threshold: The threshold that was applied.
        ai_verdict: Human-readable verdict string.
        ai_model_mode: Mode of the classifier (``"fine-tuned"`` or
            ``"heuristic"``).
        error: If report generation encountered a recoverable error (e.g.
            the image could not be opened), the error message is stored here
            and all other fields may be ``None``.  ``None`` on success.
    """

    source: str
    is_remote: bool
    origin: str
    alt_text: Optional[str]
    extension: str
    width: Optional[int]
    height: Optional[int]
    aspect_ratio: Optional[float]
    file_size_bytes: Optional[int]
    format: Optional[str]
    mode: Optional[str]
    colour_stats: Optional[ColourStats]
    ai_score: Optional[float]
    ai_is_flagged: Optional[bool]
    ai_threshold: Optional[float]
    ai_verdict: Optional[str]
    ai_model_mode: Optional[str]
    error: Optional[str] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a JSON-compatible dictionary.

        Returns:
            A flat dictionary suitable for JSON serialisation.
        """
        return {
            "source": self.source,
            "is_remote": self.is_remote,
            "origin": self.origin,
            "alt_text": self.alt_text,
            "extension": self.extension,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "file_size_bytes": self.file_size_bytes,
            "format": self.format,
            "mode": self.mode,
            "colour_stats": (
                self.colour_stats.to_dict() if self.colour_stats is not None else None
            ),
            "ai_score": (
                round(self.ai_score, 4) if self.ai_score is not None else None
            ),
            "ai_is_flagged": self.ai_is_flagged,
            "ai_threshold": self.ai_threshold,
            "ai_verdict": self.ai_verdict,
            "ai_model_mode": self.ai_model_mode,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    ref: ImageRef,
    classification: ClassificationResult,
    image: Image.Image,
    *,
    palette_colours: int = DEFAULT_PALETTE_COLOURS,
) -> ImageReport:
    """Generate a metadata report for an image.

    Combines the image reference metadata, visual statistics extracted from
    the PIL image, and the AI classification result into a single
    :class:`ImageReport`.

    Args:
        ref: The :class:`~ai_image_audit.scanner.ImageRef` describing where
            the image came from.
        classification: The :class:`~ai_image_audit.classifier.ClassificationResult`
            produced by running the classifier on this image.
        image: The opened :class:`PIL.Image.Image` instance.  The image is
            not closed or modified by this function.
        palette_colours: Number of dominant colours to extract.  Defaults to
            :data:`DEFAULT_PALETTE_COLOURS`.

    Returns:
        A fully populated :class:`ImageReport`.

    Raises:
        TypeError: If *image* is not a :class:`PIL.Image.Image`.
        TypeError: If *classification* is not a :class:`ClassificationResult`.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"generate_report() expects a PIL.Image.Image, "
            f"got {type(image).__name__!r}."
        )
    if not isinstance(classification, ClassificationResult):
        raise TypeError(
            f"generate_report() expects a ClassificationResult, "
            f"got {type(classification).__name__!r}."
        )

    width, height = image.size
    aspect_ratio: Optional[float] = (
        round(width / height, 4) if height > 0 else None
    )

    file_size = _get_file_size(ref)
    img_format = image.format or "UNKNOWN"
    img_mode = image.mode

    try:
        colour_stats = compute_colour_stats(image, n_colours=palette_colours)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not compute colour stats for %s: %s", ref.source, exc
        )
        colour_stats = None

    return ImageReport(
        source=ref.source,
        is_remote=ref.is_remote,
        origin=ref.origin,
        alt_text=ref.alt_text,
        extension=ref.extension,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        file_size_bytes=file_size,
        format=img_format,
        mode=img_mode,
        colour_stats=colour_stats,
        ai_score=classification.score,
        ai_is_flagged=classification.is_flagged,
        ai_threshold=classification.threshold,
        ai_verdict=classification.verdict,
        ai_model_mode=classification.model_mode,
        error=None,
    )


def generate_report_from_path(
    ref: ImageRef,
    classification: ClassificationResult,
    *,
    palette_colours: int = DEFAULT_PALETTE_COLOURS,
) -> ImageReport:
    """Generate a metadata report by opening the image from the ref's source path.

    A convenience wrapper around :func:`generate_report` for local images.
    Handles missing and corrupt files by returning an error report instead of
    raising an exception.

    Args:
        ref: The :class:`~ai_image_audit.scanner.ImageRef`.  Must be a local
            (non-remote) reference with a valid filesystem path.
        classification: The classification result for this image.
        palette_colours: Number of dominant colours to extract.

    Returns:
        An :class:`ImageReport`.  If the image cannot be opened, the report
        will have ``error`` set to a description of the problem and all
        image-specific fields will be ``None``.
    """
    try:
        with Image.open(ref.source) as img:
            img.load()
            return generate_report(
                ref,
                classification,
                img,
                palette_colours=palette_colours,
            )
    except FileNotFoundError as exc:
        return _error_report(ref, classification, f"File not found: {exc}")
    except UnidentifiedImageError as exc:
        return _error_report(
            ref, classification, f"Cannot identify image (corrupt or unsupported): {exc}"
        )
    except OSError as exc:
        return _error_report(ref, classification, f"OS error opening image: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _error_report(ref, classification, f"Unexpected error: {exc}")


def compute_colour_stats(
    image: Image.Image,
    *,
    n_colours: int = DEFAULT_PALETTE_COLOURS,
) -> ColourStats:
    """Compute per-channel statistics and dominant colours for a PIL image.

    The image is converted to RGB before analysis.  Channel statistics
    (mean, standard deviation) are computed using :mod:`numpy`.  Dominant
    colours are extracted by quantising the image to a small palette and
    selecting the most frequently occurring colours.

    Args:
        image: Source :class:`PIL.Image.Image` in any mode.
        n_colours: Number of dominant colours to return.  Must be >= 1.

    Returns:
        A :class:`ColourStats` instance.

    Raises:
        ValueError: If *n_colours* is less than 1.
        TypeError: If *image* is not a PIL image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"compute_colour_stats() expects a PIL.Image.Image, "
            f"got {type(image).__name__!r}."
        )
    if n_colours < 1:
        raise ValueError(f"n_colours must be >= 1, got {n_colours!r}.")

    # Convert to RGB for uniform analysis.
    rgb = _to_rgb_safe(image)

    # Pixel array of shape (H * W, 3).
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)

    mean_r, mean_g, mean_b = float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())
    std_r, std_g, std_b = float(arr[:, 0].std()), float(arr[:, 1].std()), float(arr[:, 2].std())

    dominant = _extract_dominant_colours(rgb, n_colours=n_colours)

    return ColourStats(
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
        std_r=std_r,
        std_g=std_g,
        std_b=std_b,
        dominant_colours=dominant,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_file_size(ref: ImageRef) -> Optional[int]:
    """Attempt to determine the file size of the image referenced by *ref*.

    For local files the size is read from the filesystem.  For remote images
    the size is not available without re-downloading, so ``None`` is returned.

    Args:
        ref: The image reference.

    Returns:
        File size in bytes, or ``None`` if unavailable.
    """
    if ref.is_remote:
        return None
    try:
        return os.path.getsize(ref.source)
    except OSError:
        return None


def _to_rgb_safe(image: Image.Image) -> Image.Image:
    """Convert *image* to RGB, compositing transparency over white if needed.

    Args:
        image: Source PIL image in any mode.

    Returns:
        A PIL image in ``"RGB"`` mode.
    """
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "LA":
            image = image.convert("RGBA")
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def _extract_dominant_colours(
    rgb_image: Image.Image,
    *,
    n_colours: int,
) -> List[Tuple[int, int, int]]:
    """Extract the *n_colours* most dominant colours from *rgb_image*.

    Uses PIL's built-in palette quantisation (median-cut algorithm) to
    reduce the image to a small colour palette, then returns the palette
    colours ordered by approximate frequency (most frequent first).

    Args:
        rgb_image: A PIL image already in ``"RGB"`` mode.
        n_colours: How many dominant colours to return.

    Returns:
        A list of ``(R, G, B)`` tuples, length at most *n_colours*.
    """
    # Ensure minimum image size for quantisation to work reliably.
    if rgb_image.width < 1 or rgb_image.height < 1:
        return []

    quantise_count = max(n_colours, _QUANTISE_COLOURS)

    try:
        # Convert via QUANTIZE to get a palette image.
        quantised = rgb_image.quantize(colors=quantise_count, method=Image.Quantize.MEDIANCUT)
    except Exception:  # noqa: BLE001 – fallback for very small images
        try:
            quantised = rgb_image.quantize(colors=max(1, n_colours))
        except Exception:  # noqa: BLE001
            # Last resort: just return the mean colour.
            arr = np.asarray(rgb_image, dtype=np.uint8)
            mean = tuple(int(arr[:, :, c].mean()) for c in range(3))
            return [mean]  # type: ignore[list-item]

    # Count pixel frequencies per palette index.
    pixel_array = np.asarray(quantised, dtype=np.int32).flatten()
    counts = np.bincount(pixel_array, minlength=quantise_count)

    # Get palette colours in RGB order (PIL palette is a flat byte list: R,G,B,...)
    palette_bytes = quantised.getpalette()  # list of ints [R0,G0,B0, R1,G1,B1, ...]
    if palette_bytes is None:
        return []

    # Build colour list sorted by descending frequency.
    indexed_counts = sorted(
        enumerate(counts), key=lambda x: x[1], reverse=True
    )

    dominant: List[Tuple[int, int, int]] = []
    for idx, _count in indexed_counts[:n_colours]:
        base = idx * 3
        if base + 2 < len(palette_bytes):
            r, g, b = palette_bytes[base], palette_bytes[base + 1], palette_bytes[base + 2]
            dominant.append((int(r), int(g), int(b)))

    return dominant


def _error_report(
    ref: ImageRef,
    classification: ClassificationResult,
    error_message: str,
) -> ImageReport:
    """Build an error :class:`ImageReport` when the image cannot be processed.

    Args:
        ref: The image reference that triggered the error.
        classification: The classification result (may still be valid).
        error_message: Human-readable description of the error.

    Returns:
        An :class:`ImageReport` with all image-specific fields set to
        ``None`` and *error* populated.
    """
    logger.warning("Error generating report for %s: %s", ref.source, error_message)
    return ImageReport(
        source=ref.source,
        is_remote=ref.is_remote,
        origin=ref.origin,
        alt_text=ref.alt_text,
        extension=ref.extension,
        width=None,
        height=None,
        aspect_ratio=None,
        file_size_bytes=None,
        format=None,
        mode=None,
        colour_stats=None,
        ai_score=classification.score if classification is not None else None,
        ai_is_flagged=classification.is_flagged if classification is not None else None,
        ai_threshold=classification.threshold if classification is not None else None,
        ai_verdict=classification.verdict if classification is not None else None,
        ai_model_mode=classification.model_mode if classification is not None else None,
        error=error_message,
    )
