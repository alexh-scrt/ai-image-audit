# AI Image Audit

A web tool that scans a local image directory or website URL, classifies each image asset using a lightweight CNN-based classifier to flag likely AI-generated content, generates a metadata report per flagged image, and suggests royalty-free human-made alternatives via the [Openverse](https://openverse.org) API.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the Application](#running-the-application)
8. [API Reference](#api-reference)
   - [GET /api/health](#get-apihealth)
   - [POST /api/scan](#post-apiscan)
   - [GET /api/report/\<job\_id\>](#get-apireportjob_id)
   - [GET /api/suggestions](#get-apisuggestions)
9. [Frontend Dashboard](#frontend-dashboard)
10. [Module Reference](#module-reference)
    - [ai_image_audit.scanner](#ai_image_auditscanner)
    - [ai_image_audit.classifier](#ai_image_auditclassifier)
    - [ai_image_audit.report](#ai_image_auditreport)
    - [ai_image_audit.suggester](#ai_image_auditsuggester)
11. [Running Tests](#running-tests)
12. [Project Structure](#project-structure)
13. [Limitations and Known Issues](#limitations-and-known-issues)
14. [Contributing](#contributing)
15. [License](#license)

---

## Overview

AI Image Audit was inspired by real-world controversies around studios needing to audit and replace AI-generated art in shipped products. It enables teams and developers to:

- **Discover** all image assets in a local project directory or on a live website automatically.
- **Classify** each image on a 0–1 AI-likelihood scale using a MobileNetV2-backed classifier.
- **Report** rich per-image metadata including dimensions, colour palette, file size, and AI confidence score.
- **Suggest** Creative Commons–licensed, royalty-free human-made replacement images sourced from the Openverse API.

All of this is accessible through a clean single-page web dashboard as well as a JSON REST API.

---

## Features

| Feature | Description |
|---|---|
| **Dual-mode scanning** | Accepts a local filesystem directory path *or* a public website URL and collects all discoverable image assets automatically. |
| **CNN-based classifier** | Lightweight MobileNetV2 backbone scores each image on a 0–1 AI-likelihood scale. Supports a configurable threshold for flagging. Ships with a heuristic proxy mode (no fine-tuned weights required) and a fine-tuned mode when a `.pt` checkpoint is supplied. |
| **Per-image metadata reports** | Dimensions, file size, image format/mode, dominant colour palette (up to 5 swatches), per-channel mean/std, aspect ratio, and AI confidence score. |
| **Royalty-free alternatives** | Queries the Openverse open image API with auto-derived search terms and returns Creative Commons–licensed human-made alternatives. |
| **Single-page dashboard** | Displays scan progress, flagged image cards with overlay badges, expandable metadata panels, and a side-by-side suggestion carousel for each flagged asset. |
| **REST API** | All functionality is exposed as a JSON API — integrate into CI pipelines or external tooling. |
| **Comprehensive test suite** | Unit and integration tests for all modules using `pytest`, `responses`, and synthetic images. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                              │
│   index.html + app.js + style.css (single-page dashboard)   │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP (JSON)
┌───────────────────────────▼─────────────────────────────────┐
│                    Flask API (app.py)                        │
│  POST /api/scan · GET /api/report/<id> · GET /api/suggestions│
└────┬──────────────┬───────────────┬───────────────┬──────────┘
     │              │               │               │
┌────▼────┐  ┌──────▼──────┐  ┌────▼────┐  ┌──────▼──────┐
│ scanner │  │  classifier │  │ report  │  │  suggester  │
│  .py    │  │    .py      │  │  .py    │  │    .py      │
└────┬────┘  └──────┬──────┘  └─────────┘  └──────┬──────┘
     │              │                              │
  local FS    MobileNetV2                   Openverse API
  or URL      (PyTorch)                    (openverse.org)
```

Scan jobs are processed **synchronously** and stored in an in-memory dictionary keyed by a UUID. This is appropriate for development and small-scale audits. For larger deployments, the job store can be replaced with a persistent backend (Redis, SQLite, etc.).

---

## Requirements

- Python **3.9** or later
- pip
- Internet access (for website URL scanning and Openverse suggestions)

Python dependencies (see `requirements.txt`):

```
flask>=3.0.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
numpy>=1.24.0

# Dev / test only
pytest>=7.4.0
responses>=0.23.0
```

> **Note:** PyTorch can be large (~700 MB for CPU-only). For CI or Docker environments consider the `torch` CPU-only wheel: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/ai-image-audit.git
cd ai-image-audit
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install the package in editable mode (also installs runtime deps):

```bash
pip install -e .
```

To include development/test extras:

```bash
pip install -e ".[dev]"
```

---

## Configuration

All configuration is handled through **environment variables** or by passing a config dictionary to `create_app()` directly.

| Environment Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev-secret-key` | Flask secret key. **Change this in production.** |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode (`true`/`false`). |
| `FLASK_HOST` | `127.0.0.1` | Host to bind the development server to. |
| `FLASK_PORT` | `5000` | Port for the development server. |
| `AI_FLAG_THRESHOLD` | `0.5` | Default probability threshold (0–1) above which images are flagged as AI-generated. |
| `MAX_IMAGES_PER_JOB` | `200` | Maximum number of images processed per scan job. |
| `SUGGESTIONS_PER_IMAGE` | `5` | Default number of Openverse alternatives returned per query. |
| `CLASSIFIER_MODEL_PATH` | *(none)* | Path to a fine-tuned MobileNetV2 `.pt` checkpoint. When unset, heuristic mode is used. |
| `CLASSIFIER_DEVICE` | `cpu` | Torch device (`cpu`, `cuda`, `mps`). |
| `OPENVERSE_API_BASE` | `https://api.openverse.org/v1` | Openverse API base URL (override for testing). |

### Example `.env` file

```dotenv
SECRET_KEY=my-very-secret-production-key
FLASK_DEBUG=false
AI_FLAG_THRESHOLD=0.6
CLASSIFIER_DEVICE=cpu
CLASSIFIER_MODEL_PATH=/models/ai_detector_v2.pt
```

---

## Running the Application

### Development server

```bash
python -m ai_image_audit.app
```

Or using the installed script entry-point:

```bash
ai-image-audit
```

Or with Flask's built-in CLI:

```bash
flask --app ai_image_audit.app:create_app run --reload
```

The dashboard will be available at **http://127.0.0.1:5000/**.

### Production (Gunicorn)

```bash
pip install gunicorn
gunicorn "ai_image_audit:create_app()" -w 2 -b 0.0.0.0:8000
```

### Docker (example)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "ai_image_audit:create_app()", "-w", "2", "-b", "0.0.0.0:8000"]
```

```bash
docker build -t ai-image-audit .
docker run -p 8000:8000 -e SECRET_KEY=changeme ai-image-audit
```

---

## API Reference

All endpoints return JSON. Errors always include `"error"` and `"message"` keys.

### GET /api/health

Simple liveness check.

**Response 200:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

---

### POST /api/scan

Start a synchronous audit scan. Discovers all image assets at the target, classifies each one, generates metadata reports, and stores results under a new job UUID.

**Request body (JSON):**

```json
{
  "target": "/home/user/project/assets",
  "threshold": 0.5
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `target` | `string` | ✅ | Local directory path or HTTP/HTTPS website URL. |
| `threshold` | `float` | ❌ | AI-flag threshold (0–1). Defaults to `AI_FLAG_THRESHOLD` config value. |

**Response 200:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "complete",
  "target": "/home/user/project/assets",
  "threshold": 0.5,
  "total": 12,
  "flagged_count": 3,
  "images": [
    {
      "source": "/home/user/project/assets/hero.jpg",
      "is_remote": false,
      "origin": "/home/user/project/assets",
      "alt_text": null,
      "extension": ".jpg",
      "width": 1920,
      "height": 1080,
      "aspect_ratio": 1.7778,
      "file_size_bytes": 524288,
      "format": "JPEG",
      "mode": "RGB",
      "colour_stats": {
        "mean_r": 142.3,
        "mean_g": 118.7,
        "mean_b": 99.2,
        "std_r": 45.1,
        "std_g": 38.4,
        "std_b": 32.9,
        "dominant_colours": [[180, 140, 110], [90, 70, 60], [220, 200, 180]]
      },
      "ai_score": 0.7812,
      "ai_is_flagged": true,
      "ai_threshold": 0.5,
      "ai_verdict": "AI-generated (flagged)",
      "ai_model_mode": "heuristic",
      "error": null
    }
  ]
}
```

**Error responses:**

| Code | Cause |
|---|---|
| `400` | Missing/empty `target`, invalid `threshold`, non-existent directory. |
| `500` | Classifier initialisation failure or unexpected processing error. |

---

### GET /api/report/\<job\_id\>

Retrieve a previously completed scan report by its job UUID.

**Response 200:** Same structure as the scan response body.

**Response 404:**
```json
{
  "error": "not_found",
  "message": "No report found for job_id 'abc-123'.",
  "job_id": "abc-123"
}
```

---

### GET /api/suggestions

Fetch royalty-free Creative Commons–licensed image alternatives from the Openverse API.

**Query parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `q` | `string` | ✅ | Search query (e.g. derived from image alt-text or filename). |
| `per_page` | `integer` | ❌ | Number of results (default: `SUGGESTIONS_PER_IMAGE` config, max 20). |

**Response 200:**

```json
{
  "query": "forest landscape photography",
  "per_page": 5,
  "total": 5,
  "suggestions": [
    {
      "id": "abc123",
      "title": "Morning Mist in the Forest",
      "url": "https://live.staticflickr.com/...",
      "thumbnail": "https://live.staticflickr.com/.../thumb.jpg",
      "foreign_landing_url": "https://www.flickr.com/photos/.../",
      "creator": "Jane Photographer",
      "creator_url": "https://www.flickr.com/people/janephoto/",
      "licence": "by",
      "licence_version": "2.0",
      "licence_url": "https://creativecommons.org/licenses/by/2.0/",
      "provider": "flickr",
      "source": "flickr",
      "tags": ["forest", "nature", "mist", "morning"],
      "width": 2048,
      "height": 1365
    }
  ]
}
```

**Error responses:**

| Code | Cause |
|---|---|
| `400` | Missing or empty `q` parameter; invalid `per_page`. |
| `502` | Upstream Openverse API returned an error or is unreachable. |

---

## Frontend Dashboard

Open **http://127.0.0.1:5000/** in your browser after starting the server.

### Workflow

1. **Enter a target** – type a local directory path (e.g. `/home/user/project/assets`) or a website URL (e.g. `https://example.com`) into the input field.
2. **Set a threshold** – adjust the AI-flag sensitivity (0.0 = flag everything, 1.0 = flag nothing, default 0.5).
3. **Run Scan** – click the button. Results appear once the synchronous scan completes.
4. **Review the summary bar** – see total images scanned, flagged count, and safe count at a glance.
5. **Filter results** – use the filter buttons to view All / Flagged only / Human-made only.
6. **Expand metadata** – click **Details** on any card to see dimensions, colour palette swatches, AI score, and more.
7. **Find Alternatives** – click **Find Alternatives** on any card to open the suggestion modal and browse royalty-free replacements from Openverse.
8. **Clear** – click the Clear button to reset and start a new scan.

### Badges

| Badge | Meaning |
|---|---|
| 🔴 **AI-generated** | Score ≥ threshold — likely AI-generated. |
| 🟢 **Human-made** | Score < threshold — likely human-made. |
| 🟡 **Error** | Image could not be opened or classified. |

---

## Module Reference

### ai_image_audit.scanner

Dual-mode image discovery.

```python
from ai_image_audit.scanner import scan, scan_directory, scan_url, ImageRef

# Auto-detect mode:
refs = scan("/path/to/images")           # local directory
refs = scan("https://example.com")       # website URL

# Explicit modes:
refs = scan_directory("/path/to/images", max_images=100)
refs = scan_url("https://example.com", max_images=50)

for ref in refs:
    print(ref.source, ref.is_remote, ref.extension, ref.alt_text)
```

**`ImageRef` fields:** `source`, `is_remote`, `origin`, `alt_text`, `extension`

**Supported extensions:** `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif`, `.svg`

The directory scanner skips: `node_modules`, `.git`, `__pycache__`, `venv`, `.venv`, `dist`, `build`, `.tox`, `.mypy_cache`, `.pytest_cache`, and any hidden directory (name starts with `.`).

The URL scanner extracts images from `<img src>`, `<img srcset>`, `<source srcset>` (inside `<picture>`), and `<meta property="og:image">` tags.

---

### ai_image_audit.classifier

MobileNetV2-backed AI-likelihood scorer.

```python
from PIL import Image
from ai_image_audit.classifier import AIImageClassifier, classify_image

# Heuristic mode (no checkpoint needed):
classifier = AIImageClassifier(threshold=0.5, device="cpu")
image = Image.open("artwork.jpg")
result = classifier.classify(image)
print(result.score, result.is_flagged, result.verdict)
# → 0.7231  True  "AI-generated (flagged)"

# Classify from a file path:
result = classifier.classify_path("/path/to/image.png")

# Fine-tuned checkpoint mode:
classifier = AIImageClassifier(
    model_path="/models/ai_detector.pt",
    threshold=0.6,
    device="cpu",
)

# Module-level convenience function:
result = classify_image(image, threshold=0.5, device="cpu")
```

**`ClassificationResult` fields:** `score` (float 0–1), `is_flagged` (bool), `threshold` (float), `verdict` (str), `model_mode` (str: `"heuristic"` or `"fine-tuned"`)

#### Classifier modes

| Mode | When active | Description |
|---|---|---|
| **heuristic** | No `model_path` supplied (default) | Uses ImageNet-pretrained MobileNetV2. Computes a proxy score from softmax entropy + art-class activations. Suitable for demonstration; accuracy is limited. |
| **fine-tuned** | `model_path` points to a valid `.pt` file | Loads a 2-class MobileNetV2 state-dict. Returns `softmax[AI_class]` as the score. Use this in production for accurate detection. |

#### Supplying a fine-tuned checkpoint

The checkpoint must be a `torch.save(model.state_dict(), ...)` of a MobileNetV2 with a 2-class classifier head:

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
# ... fine-tune on your AI-vs-human dataset ...
torch.save(model.state_dict(), "ai_detector.pt")
```

Then set `CLASSIFIER_MODEL_PATH=/path/to/ai_detector.pt` or pass `model_path=` to the constructor.

---

### ai_image_audit.report

Per-image metadata report generation.

```python
from PIL import Image
from ai_image_audit.scanner import ImageRef
from ai_image_audit.classifier import AIImageClassifier
from ai_image_audit.report import generate_report, generate_report_from_path, compute_colour_stats

ref = ImageRef(source="/assets/hero.jpg", is_remote=False, origin="/assets")
classifier = AIImageClassifier(device="cpu")

# From an open PIL image:
with Image.open(ref.source) as img:
    classification = classifier.classify(img)
    report = generate_report(ref, classification, img)

print(report.width, report.height, report.ai_score)
print(report.colour_stats.dominant_colours)
print(report.to_dict())  # JSON-serialisable dict

# From a file path (handles missing/corrupt files gracefully):
report = generate_report_from_path(ref, classification)
if report.error:
    print("Error:", report.error)
```

**`ImageReport` fields:** `source`, `is_remote`, `origin`, `alt_text`, `extension`, `width`, `height`, `aspect_ratio`, `file_size_bytes`, `format`, `mode`, `colour_stats` (`ColourStats`), `ai_score`, `ai_is_flagged`, `ai_threshold`, `ai_verdict`, `ai_model_mode`, `error`

**`ColourStats` fields:** `mean_r/g/b`, `std_r/g/b`, `dominant_colours` (list of RGB tuples)

---

### ai_image_audit.suggester

Openverse API client for royalty-free image suggestions.

```python
from ai_image_audit.suggester import fetch_suggestions, OpenverseSuggester

# Convenience function:
suggestions = fetch_suggestions("forest landscape photography", per_page=5)
for s in suggestions:
    print(s.title, s.url, s.licence)

# Reusable instance (preferred for multiple queries):
with OpenverseSuggester(per_page=5) as suggester:
    results = suggester.suggest("mountain sunset")
    more = suggester.suggest("ocean waves", per_page=3)
```

**`ImageSuggestion` fields:** `id`, `title`, `url`, `thumbnail`, `foreign_landing_url`, `creator`, `creator_url`, `licence`, `licence_version`, `licence_url`, `provider`, `source`, `tags`, `width`, `height`

Default licence filter: `cc0,by,by-sa` (most permissive Creative Commons licences).

---

## Running Tests

All tests live in the `tests/` directory and use `pytest`.

```bash
# Install dev dependencies first:
pip install -e ".[dev]"

# Run all tests:
pytest

# Run with verbose output:
pytest -v

# Run a specific test file:
pytest tests/test_classifier.py -v

# Run a specific test class or function:
pytest tests/test_scanner.py::TestScanDirectory::test_finds_nested_images -v

# Run with coverage (requires pytest-cov):
pip install pytest-cov
pytest --cov=ai_image_audit --cov-report=term-missing
```

### Test suite overview

| File | What it tests |
|---|---|
| `tests/test_app_factory.py` | Flask app factory, config defaults, stub endpoints, error handlers. |
| `tests/test_api_routes.py` | Full integration tests for all API routes (scan, report, suggestions). |
| `tests/test_scanner.py` | Directory and URL scanning, `ImageRef`, `_is_url`, `_parse_srcset`, `_derive_base_url`. |
| `tests/test_classifier.py` | `AIImageClassifier`, `ClassificationResult`, `_to_rgb`, `_heuristic_score`, `_finetuned_score`, fine-tuned checkpoint loading. |
| `tests/test_report.py` | `ImageReport`, `ColourStats`, `compute_colour_stats`, `generate_report`, `generate_report_from_path`, error reports. |
| `tests/test_suggester.py` | `OpenverseSuggester`, `ImageSuggestion`, `_parse_suggestion`, `fetch_suggestions`, `_build_url`, context manager. |

All external HTTP calls (Openverse API, remote image downloads) are **mocked** using the `responses` library so tests run fully offline and deterministically.

---

## Project Structure

```
ai-image-audit/
├── pyproject.toml              # Project metadata and build config
├── requirements.txt            # Pinned pip dependencies
├── README.md                   # This file
│
├── ai_image_audit/
│   ├── __init__.py             # Package init — exposes create_app()
│   ├── app.py                  # Flask app factory + all API routes
│   ├── classifier.py           # MobileNetV2-backed AI image classifier
│   ├── scanner.py              # Local directory + URL image scanner
│   ├── report.py               # Per-image metadata report generation
│   ├── suggester.py            # Openverse API client for suggestions
│   └── static/
│       ├── index.html          # Single-page dashboard
│       ├── app.js              # Vanilla JS frontend logic
│       └── style.css           # Dashboard styles
│
└── tests/
    ├── __init__.py
    ├── test_app_factory.py
    ├── test_api_routes.py
    ├── test_scanner.py
    ├── test_classifier.py
    ├── test_report.py
    └── test_suggester.py
```

---

## Limitations and Known Issues

### Classifier accuracy in heuristic mode

The default **heuristic mode** uses an ImageNet-pretrained MobileNetV2 without any fine-tuning on an AI-vs-human image dataset. It computes a proxy score from softmax entropy and art-class activations. This approach:

- Will produce many **false positives** (human art flagged as AI) and **false negatives** (AI images not flagged).
- Is provided so the tool works out-of-the-box for **demonstration and integration testing**.
- Should **not** be used for production auditing where accuracy matters.

For accurate detection, supply a fine-tuned checkpoint via `CLASSIFIER_MODEL_PATH`. Several public datasets exist for training such a classifier (e.g. datasets from academic papers on AI image detection).

### In-memory job store

Scan job results are stored in a Python dictionary in RAM. This means:

- Jobs are **lost on server restart**.
- Memory usage grows with the number of scans (no eviction policy).
- Not suitable for multi-worker deployments (each worker has its own store).

For production use, replace `_JOB_STORE` in `app.py` with a Redis, SQLite, or PostgreSQL-backed store.

### Synchronous scan processing

Scans run synchronously in the request handler. Large scans (many images or slow remote URLs) will block the server thread for the duration of the scan. For production use, consider offloading scans to a background worker (Celery, RQ, etc.) and polling for results.

### Remote image scanning

- Only images linked directly from the HTML of the target URL are discovered (no deep crawling).
- Dynamic images loaded via JavaScript (after page render) are not detected.
- Some websites block automated HTTP clients; the scanner sends a browser-like User-Agent but does not support JavaScript rendering.

### SVG files

SVG files are discovered by the scanner but cannot be classified by the CNN (SVG is a vector format). The classifier will attempt to open SVGs with PIL; if PIL cannot parse them, an error report is generated.

### Openverse API rate limits

The Openverse API is rate-limited. Heavy usage may result in `429 Too Many Requests` responses surfaced as 502 errors from the `/api/suggestions` endpoint. Consider caching suggestion results.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository and create a feature branch.
2. Ensure all existing tests pass: `pytest`.
3. Add tests for any new functionality.
4. Follow PEP 8 style conventions (the project uses type hints throughout).
5. Open a pull request with a clear description of the changes.

### Development setup

```bash
git clone https://github.com/your-org/ai-image-audit.git
cd ai-image-audit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest  # verify everything passes
```

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 AI Image Audit Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

*Built with [Flask](https://flask.palletsprojects.com/), [PyTorch](https://pytorch.org/), [Pillow](https://python-pillow.org/), and [Openverse](https://openverse.org).*
