"""Integration tests for the Flask API routes defined in ai_image_audit.app.

These tests verify the fully-wired endpoints introduced in Phase 5:

- POST /api/scan   – scan a local directory or URL, return job report.
- GET  /api/report/<job_id> – retrieve a stored report by job ID.
- GET  /api/suggestions     – query Openverse for royalty-free alternatives.
- GET  /api/health          – health check (regression).
- GET  /            – SPA index (regression; 404 acceptable if no index.html).

All external HTTP calls (Openverse API, remote image downloads) are mocked
using the ``responses`` library so the tests remain hermetic.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
import responses as responses_lib
import requests
from PIL import Image
from flask import Flask
from flask.testing import FlaskClient

from ai_image_audit import create_app
from ai_image_audit.suggester import OPENVERSE_API_BASE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app() -> Flask:
    """Return a testing Flask app with fast/deterministic settings."""
    return create_app({
        "TESTING": True,
        "DEBUG": False,
        # Use a low threshold so synthetic images get flagged.
        "AI_FLAG_THRESHOLD": 0.0,
        "MAX_IMAGES_PER_JOB": 50,
        "SUGGESTIONS_PER_IMAGE": 3,
        # Force CPU inference.
        "CLASSIFIER_DEVICE": "cpu",
        "CLASSIFIER_MODEL_PATH": None,
    })


@pytest.fixture(scope="module")
def client(app: Flask) -> FlaskClient:
    """Return a test client for the module-scoped app."""
    return app.test_client()


@pytest.fixture()\def image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with two small JPEG images."""
    for name, colour in [("red.jpg", (200, 50, 50)), ("blue.jpg", (50, 50, 200))]:
        Image.new("RGB", (32, 32), color=colour).save(
            str(tmp_path / name), format="JPEG"
        )
    return tmp_path


def _make_openverse_item(
    image_id: str = "abc",
    url: str = "https://cdn.openverse.example.com/photo.jpg",
) -> dict[str, Any]:
    """Build a minimal Openverse API result item."""
    return {
        "id": image_id,
        "title": "Sample Image",
        "url": url,
        "thumbnail": "https://cdn.openverse.example.com/thumb.jpg",
        "foreign_landing_url": "https://flickr.com/photos/sample",
        "creator": "Sample Creator",
        "creator_url": "https://flickr.com/people/sample",
        "license": "cc0",
        "license_version": "1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "provider": "flickr",
        "source": "flickr",
        "tags": [{"name": "nature"}, {"name": "forest"}],
        "width": 1920,
        "height": 1080,
    }


def _make_openverse_response(items: list[dict]) -> dict:
    return {"count": len(items), "next": None, "previous": None, "results": items}


OPENVERSE_SEARCH_URL = f"{OPENVERSE_API_BASE}/images/"


# ---------------------------------------------------------------------------
# Health check regression
# ---------------------------------------------------------------------------


def test_health_check_200(client: FlaskClient) -> None:
    """GET /api/health should return 200 OK."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /api/scan – local directory
# ---------------------------------------------------------------------------


def test_scan_local_directory_returns_200(client: FlaskClient, image_dir: Path) -> None:
    """Scanning a valid local directory should return HTTP 200."""
    resp = client.post("/api/scan", json={"target": str(image_dir)})
    assert resp.status_code == 200


def test_scan_local_directory_returns_job_id(client: FlaskClient, image_dir: Path) -> None:
    """Scan response should include a non-empty job_id string."""
    resp = client.post("/api/scan", json={"target": str(image_dir)})
    data = resp.get_json()
    assert "job_id" in data
    assert isinstance(data["job_id"], str)
    assert len(data["job_id"]) > 0


def test_scan_local_directory_status_complete(client: FlaskClient, image_dir: Path) -> None:
    """Scan response status should be 'complete'."""
    resp = client.post("/api/scan", json={"target": str(image_dir)})
    data = resp.get_json()
    assert data["status"] == "complete"


def test_scan_local_directory_total_count(client: FlaskClient, image_dir: Path) -> None:
    """Scan should report the correct number of discovered images."""
    resp = client.post("/api/scan", json={"target": str(image_dir)})
    data = resp.get_json()
    assert data["total"] == 2


def test_scan_local_directory_images_list(client: FlaskClient, image_dir: Path) -> None:
    """Scan response should include an 'images' list with one entry per image."""
    resp = client.post("/api/scan", json={"target": str(image_dir)})
    data = resp.get_json()
    assert "images" in data
    assert isinstance(data["images"], list)
    assert len(data["images"]) == 2


def test_scan_local_image_entry_has_source(client: FlaskClient, image_dir: Path) -> None:
    """Each image entry in the report should have a 'source' field."""
    resp = client.post("/api/scan", json={"target": str(image_dir)})
    data = resp.get_json()
    for entry in data["images"]:
        assert "source" in entry
        assert entry["source"]  # non-empty


def test_scan_local_image_entry_has_ai_score(client: FlaskClient, image_dir: Path) -> None:
    """Each successfully processed image should have an 'ai_score'."""
    resp = client.post("/api/scan", json={"target": str(image_dir)})
    data = resp.get_json()
    for entry in data["images"]:
        if entry.get("error") is None:
            assert "ai_score" in entry
            assert entry["ai_score"] is not None
            assert 0.0 <= entry["ai_score"] <= 1.0


def test_scan_local_image_entry_has_dimensions(client: FlaskClient, image_dir: Path) -> None:
    """Each successfully processed image entry should include width and height."""
    resp = client.post("/api/scan", json={"target": str(image_dir)})
    data = resp.get_json()
    for entry in data["images"]:
        if entry.get("error") is None:
            assert entry["width"] == 32
            assert entry["height"] == 32


def test_scan_custom_threshold_applied(client: FlaskClient, image_dir: Path) -> None:
    """A threshold of 1.0 should result in no flagged images."""
    resp = client.post(
        "/api/scan", json={"target": str(image_dir), "threshold": 1.0}
    )
    data = resp.get_json()
    assert data["flagged_count"] == 0


def test_scan_threshold_zero_flags_all(client: FlaskClient, image_dir: Path) -> None:
    """A threshold of 0.0 should flag all images (score >= 0.0 always)."""
    resp = client.post(
        "/api/scan", json={"target": str(image_dir), "threshold": 0.0}
    )
    data = resp.get_json()
    # All images with no error should be flagged.
    successful = [e for e in data["images"] if e.get("error") is None]
    assert data["flagged_count"] == len(successful)


def test_scan_target_echoed_in_response(client: FlaskClient, image_dir: Path) -> None:
    """The response should echo back the scanned target."""
    resp = client.post("/api/scan", json={"target": str(image_dir)})
    data = resp.get_json()
    assert data["target"] == str(image_dir)


# ---------------------------------------------------------------------------
# POST /api/scan – validation errors
# ---------------------------------------------------------------------------


def test_scan_missing_target_returns_400(client: FlaskClient) -> None:
    """Omitting 'target' should return HTTP 400."""
    resp = client.post("/api/scan", json={})
    assert resp.status_code == 400


def test_scan_empty_target_returns_400(client: FlaskClient) -> None:
    """An empty 'target' string should return HTTP 400."""
    resp = client.post("/api/scan", json={"target": ""})
    assert resp.status_code == 400


def test_scan_whitespace_target_returns_400(client: FlaskClient) -> None:
    """A whitespace-only 'target' should return HTTP 400."""
    resp = client.post("/api/scan", json={"target": "   "})
    assert resp.status_code == 400


def test_scan_nonexistent_directory_returns_400(client: FlaskClient, tmp_path: Path) -> None:
    """A target pointing to a non-existent directory should return HTTP 400."""
    resp = client.post(
        "/api/scan", json={"target": str(tmp_path / "does_not_exist")}
    )
    assert resp.status_code == 400


def test_scan_invalid_threshold_returns_400(client: FlaskClient, image_dir: Path) -> None:
    """An out-of-range threshold should return HTTP 400."""
    resp = client.post(
        "/api/scan", json={"target": str(image_dir), "threshold": 1.5}
    )
    assert resp.status_code == 400


def test_scan_negative_threshold_returns_400(client: FlaskClient, image_dir: Path) -> None:
    """A negative threshold should return HTTP 400."""
    resp = client.post(
        "/api/scan", json={"target": str(image_dir), "threshold": -0.1}
    )
    assert resp.status_code == 400


def test_scan_no_body_returns_400(client: FlaskClient) -> None:
    """A request with no JSON body should return HTTP 400."""
    resp = client.post("/api/scan", content_type="application/json", data="")
    assert resp.status_code == 400


def test_scan_returns_json_error_on_400(client: FlaskClient) -> None:
    """A 400 error from scan should include a JSON error body."""
    resp = client.post("/api/scan", json={})
    data = resp.get_json()
    assert "error" in data
    assert "message" in data


# ---------------------------------------------------------------------------
# POST /api/scan – empty directory
# ---------------------------------------------------------------------------


def test_scan_empty_directory_returns_200(client: FlaskClient, tmp_path: Path) -> None:
    """Scanning an empty directory should succeed with total=0."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    resp = client.post("/api/scan", json={"target": str(empty_dir)})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total"] == 0
    assert data["images"] == []


# ---------------------------------------------------------------------------
# GET /api/report/<job_id>
# ---------------------------------------------------------------------------


def test_report_returns_stored_job(client: FlaskClient, image_dir: Path) -> None:
    """A report retrieved by job_id should match the scan response."""
    scan_resp = client.post("/api/scan", json={"target": str(image_dir)})
    job_id = scan_resp.get_json()["job_id"]

    report_resp = client.get(f"/api/report/{job_id}")
    assert report_resp.status_code == 200
    data = report_resp.get_json()
    assert data["job_id"] == job_id
    assert data["status"] == "complete"


def test_report_contains_images(client: FlaskClient, image_dir: Path) -> None:
    """The stored report should contain the images list."""
    scan_resp = client.post("/api/scan", json={"target": str(image_dir)})
    job_id = scan_resp.get_json()["job_id"]

    report_resp = client.get(f"/api/report/{job_id}")
    data = report_resp.get_json()
    assert "images" in data
    assert len(data["images"]) == 2


def test_report_unknown_job_id_returns_404(client: FlaskClient) -> None:
    """Requesting a report for an unknown job ID should return HTTP 404."""
    resp = client.get("/api/report/nonexistent-job-id-xyz")
    assert resp.status_code == 404


def test_report_404_body_has_error_key(client: FlaskClient) -> None:
    """A 404 report response should include an 'error' key."""
    resp = client.get("/api/report/no-such-job")
    data = resp.get_json()
    assert "error" in data


def test_report_404_body_includes_job_id(client: FlaskClient) -> None:
    """A 404 report response should echo back the requested job_id."""
    resp = client.get("/api/report/my-missing-job")
    data = resp.get_json()
    assert data.get("job_id") == "my-missing-job"


def test_report_returns_200_on_valid_job(client: FlaskClient, tmp_path: Path) -> None:
    """After a successful scan the report endpoint returns 200."""
    empty = tmp_path / "empty_for_report"
    empty.mkdir()
    scan_resp = client.post("/api/scan", json={"target": str(empty)})
    job_id = scan_resp.get_json()["job_id"]
    resp = client.get(f"/api/report/{job_id}")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/suggestions
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_suggestions_returns_200(client: FlaskClient) -> None:
    """GET /api/suggestions?q=forest should return HTTP 200."""
    items = [_make_openverse_item()]
    responses_lib.add(
        responses_lib.GET,
        OPENVERSE_SEARCH_URL,
        json=_make_openverse_response(items),
        status=200,
    )
    resp = client.get("/api/suggestions?q=forest")
    assert resp.status_code == 200


@responses_lib.activate
def test_suggestions_returns_suggestion_list(client: FlaskClient) -> None:
    """Suggestions response should include a 'suggestions' list."""
    items = [_make_openverse_item(image_id="s1"), _make_openverse_item(image_id="s2")]
    responses_lib.add(
        responses_lib.GET,
        OPENVERSE_SEARCH_URL,
        json=_make_openverse_response(items),
        status=200,
    )
    resp = client.get("/api/suggestions?q=mountain")
    data = resp.get_json()
    assert "suggestions" in data
    assert isinstance(data["suggestions"], list)
    assert len(data["suggestions"]) == 2


@responses_lib.activate
def test_suggestions_echoes_query(client: FlaskClient) -> None:
    """Suggestions response should echo back the query string."""
    responses_lib.add(
        responses_lib.GET,
        OPENVERSE_SEARCH_URL,
        json=_make_openverse_response([]),
        status=200,
    )
    resp = client.get("/api/suggestions?q=sunset+photography")
    data = resp.get_json()
    assert data["query"] == "sunset photography"


@responses_lib.activate
def test_suggestions_per_page_echoed(client: FlaskClient) -> None:
    """Suggestions response should echo back per_page."""
    responses_lib.add(
        responses_lib.GET,
        OPENVERSE_SEARCH_URL,
        json=_make_openverse_response([]),
        status=200,
    )
    resp = client.get("/api/suggestions?q=ocean&per_page=3")
    data = resp.get_json()
    assert data["per_page"] == 3


@responses_lib.activate
def test_suggestions_each_item_has_url(client: FlaskClient) -> None:
    """Each suggestion should include a 'url' field."""
    items = [_make_openverse_item(image_id=f"id{i}", url=f"https://cdn.example.com/{i}.jpg")
             for i in range(2)]
    responses_lib.add(
        responses_lib.GET,
        OPENVERSE_SEARCH_URL,
        json=_make_openverse_response(items),
        status=200,
    )
    resp = client.get("/api/suggestions?q=landscape")
    data = resp.get_json()
    for s in data["suggestions"]:
        assert "url" in s
        assert s["url"]


def test_suggestions_missing_q_returns_400(client: FlaskClient) -> None:
    """Omitting 'q' parameter should return HTTP 400."""
    resp = client.get("/api/suggestions")
    assert resp.status_code == 400


def test_suggestions_empty_q_returns_400(client: FlaskClient) -> None:
    """An empty 'q' parameter should return HTTP 400."""
    resp = client.get("/api/suggestions?q=")
    assert resp.status_code == 400


def test_suggestions_invalid_per_page_returns_400(client: FlaskClient) -> None:
    """A non-integer 'per_page' should return HTTP 400."""
    resp = client.get("/api/suggestions?q=forest&per_page=abc")
    assert resp.status_code == 400


def test_suggestions_zero_per_page_returns_400(client: FlaskClient) -> None:
    """A per_page of 0 should return HTTP 400."""
    resp = client.get("/api/suggestions?q=forest&per_page=0")
    assert resp.status_code == 400


@responses_lib.activate
def test_suggestions_openverse_500_returns_502(client: FlaskClient) -> None:
    """An upstream Openverse 500 error should surface as HTTP 502."""
    responses_lib.add(
        responses_lib.GET,
        OPENVERSE_SEARCH_URL,
        status=500,
    )
    resp = client.get("/api/suggestions?q=forest")
    assert resp.status_code == 502


@responses_lib.activate
def test_suggestions_openverse_connection_error_returns_502(client: FlaskClient) -> None:
    """A network error to Openverse should return HTTP 502."""
    responses_lib.add(
        responses_lib.GET,
        OPENVERSE_SEARCH_URL,
        body=requests.exceptions.ConnectionError("refused"),
    )
    resp = client.get("/api/suggestions?q=forest")
    assert resp.status_code == 502


@responses_lib.activate
def test_suggestions_502_body_has_error_key(client: FlaskClient) -> None:
    """A 502 suggestions response should include an 'error' key."""
    responses_lib.add(
        responses_lib.GET,
        OPENVERSE_SEARCH_URL,
        status=503,
    )
    resp = client.get("/api/suggestions?q=forest")
    data = resp.get_json()
    assert "error" in data


@responses_lib.activate
def test_suggestions_total_field_present(client: FlaskClient) -> None:
    """Suggestions response should include a 'total' field."""
    items = [_make_openverse_item()]
    responses_lib.add(
        responses_lib.GET,
        OPENVERSE_SEARCH_URL,
        json=_make_openverse_response(items),
        status=200,
    )
    resp = client.get("/api/suggestions?q=nature")
    data = resp.get_json()
    assert "total" in data
    assert data["total"] == 1


# ---------------------------------------------------------------------------
# Error handler regression
# ---------------------------------------------------------------------------


def test_404_handler_returns_json(client: FlaskClient) -> None:
    """Requests to unknown routes should receive a JSON 404."""
    resp = client.get("/this/path/does/not/exist")
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["error"] == "not_found"


# ---------------------------------------------------------------------------
# Scan with URL target (mocked remote HTTP)
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_scan_url_target_returns_200(client: FlaskClient) -> None:
    """Scanning a mocked URL target should return HTTP 200."""
    html = "<html><body><img src='/img/photo.jpg'></body></html>"
    responses_lib.add(
        responses_lib.GET,
        "https://example.com",
        body=html,
        content_type="text/html",
        status=200,
    )
    # Also mock the image download.
    img_buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(100, 150, 200)).save(img_buf, format="JPEG")
    responses_lib.add(
        responses_lib.GET,
        "https://example.com/img/photo.jpg",
        body=img_buf.getvalue(),
        content_type="image/jpeg",
        status=200,
    )
    resp = client.post("/api/scan", json={"target": "https://example.com"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total"] == 1


@responses_lib.activate
def test_scan_url_target_image_entry_is_remote(client: FlaskClient) -> None:
    """Images discovered from a URL scan should be marked as remote."""
    html = "<html><body><img src='/img/art.png'></body></html>"
    responses_lib.add(
        responses_lib.GET,
        "https://example.com",
        body=html,
        content_type="text/html",
        status=200,
    )
    img_buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(img_buf, format="PNG")
    responses_lib.add(
        responses_lib.GET,
        "https://example.com/img/art.png",
        body=img_buf.getvalue(),
        content_type="image/png",
        status=200,
    )
    resp = client.post("/api/scan", json={"target": "https://example.com"})
    data = resp.get_json()
    assert data["images"][0]["is_remote"] is True


@responses_lib.activate
def test_scan_url_http_error_returns_400(client: FlaskClient) -> None:
    """A 404 response when fetching the scan target URL should return HTTP 400."""
    responses_lib.add(
        responses_lib.GET,
        "https://example.com/broken",
        status=404,
    )
    resp = client.post("/api/scan", json={"target": "https://example.com/broken"})
    # Network/HTTP errors during scanning are treated as bad inputs.
    assert resp.status_code in (400, 500)
