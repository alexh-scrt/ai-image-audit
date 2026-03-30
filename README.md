# AI Image Audit

> Scan, flag, and replace AI-generated images — before they ship.

AI Image Audit is a web tool that scans a local image directory or website URL and uses a lightweight CNN classifier to detect AI-generated content. Each flagged image gets a detailed metadata report, and the tool automatically suggests royalty-free, human-made alternatives sourced from the [Openverse](https://openverse.org) API — so you can audit and remediate in one workflow.

---

## Quick Start

**Install**

```bash
# Clone the repo
git clone https://github.com/your-org/ai-image-audit.git
cd ai-image-audit

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Run the app**

```bash
flask --app ai_image_audit run --debug
```

Open your browser at [http://localhost:5000](http://localhost:5000). Enter a local directory path (e.g. `/path/to/assets`) or a public URL (e.g. `https://example.com`) and click **Scan**.

---

## Features

- **Dual-mode scanning** — accepts a local filesystem directory or a public website URL; discovers all image assets automatically, including `<img>` tags and `srcset` references.
- **CNN-based AI classifier** — MobileNetV2 scores each image on a 0–1 AI-likelihood scale with a configurable flagging threshold (default `0.5`).
- **Per-image metadata reports** — captures dimensions, file size, format, dominant colour palette, channel statistics, and the AI confidence verdict.
- **Royalty-free alternative suggestions** — queries the Openverse API with auto-derived tags and returns Creative Commons licensed, human-made replacements.
- **Clean single-page dashboard** — scan progress, flagged image cards with overlay badges, expandable metadata panels, and side-by-side suggestion carousels.

---

## Usage Examples

### Scan via the Web UI

1. Run the Flask server (`flask --app ai_image_audit run`).
2. Navigate to `http://localhost:5000`.
3. Enter a path or URL and press **Scan**.
4. Flagged images appear with an **AI** badge. Click any card to expand its metadata and browse replacement suggestions.

### Scan via the REST API

**Start a scan job**

```bash
curl -X POST http://localhost:5000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"source": "/path/to/images", "threshold": 0.6}'
```

```json
{
  "job_id": "3f7a1c2d-...",
  "flagged": 4,
  "total": 17,
  "results": [
    {
      "path": "/path/to/images/hero.png",
      "ai_score": 0.83,
      "verdict": "AI-generated",
      "dimensions": [1920, 1080],
      "format": "PNG",
      "file_size_kb": 412
    }
  ]
}
```

**Retrieve a stored report**

```bash
curl http://localhost:5000/api/report/3f7a1c2d-...
```

**Fetch royalty-free suggestions**

```bash
curl "http://localhost:5000/api/suggestions?q=forest+landscape&per_page=5"
```

```json
{
  "suggestions": [
    {
      "title": "Mountain Forest at Dusk",
      "url": "https://openverse.org/image/abc123",
      "thumbnail": "https://cdn.openverse.org/thumb/abc123.jpg",
      "license": "CC BY 2.0",
      "creator": "Jane Photographer"
    }
  ]
}
```

**Scan a public website URL**

```bash
curl -X POST http://localhost:5000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"source": "https://example.com", "threshold": 0.5}'
```

**Health check**

```bash
curl http://localhost:5000/api/health
# {"status": "ok"}
```

---

## Project Structure

```
ai-image-audit/
├── pyproject.toml                  # Project metadata and build config
├── requirements.txt                # Pinned runtime + dev dependencies
├── ai_image_audit/
│   ├── __init__.py                 # Exposes create_app factory
│   ├── app.py                      # Flask app factory + route definitions
│   ├── scanner.py                  # Directory & URL image discovery
│   ├── classifier.py               # MobileNetV2 AI-likelihood scorer
│   ├── report.py                   # Per-image metadata report generator
│   ├── suggester.py                # Openverse API alternative suggester
│   └── static/
│       ├── index.html              # Single-page dashboard UI
│       ├── app.js                  # Vanilla JS — forms, polling, rendering
│       └── style.css               # Dashboard and card styles
└── tests/
    ├── test_app_factory.py         # App factory and route smoke tests
    ├── test_api_routes.py          # Integration tests for all API endpoints
    ├── test_scanner.py             # Directory and URL scanning unit tests
    ├── test_classifier.py          # Classifier unit tests (synthetic images)
    ├── test_report.py              # Report generation unit tests
    └── test_suggester.py           # Openverse suggester unit tests
```

---

## Configuration

Configuration is passed as environment variables or overridden when calling `create_app(config)`.

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `production` | Set to `development` to enable debug mode and reloader. |
| `SECRET_KEY` | random | Flask secret key for session security. Set explicitly in production. |
| `AI_THRESHOLD` | `0.5` | Default AI-likelihood score above which an image is flagged (0.0–1.0). |
| `MODEL_PATH` | *(heuristic mode)* | Path to a fine-tuned `.pth` checkpoint file. Omit to use the built-in ImageNet heuristic. |
| `MAX_IMAGES` | `200` | Maximum number of images collected per scan job. |
| `OPENVERSE_PAGE_SIZE` | `6` | Number of alternative suggestions returned per flagged image. |

**Example — use a custom model and lower threshold:**

```bash
export MODEL_PATH=/models/mobilenet_ai_detector.pth
export AI_THRESHOLD=0.45
flask --app ai_image_audit run
```

**Run the test suite:**

```bash
pytest tests/ -v
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) — an AI agent that ships code daily.*
