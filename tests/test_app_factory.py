"""Tests for the Flask application factory defined in ai_image_audit.app.

These tests verify that:
- The factory returns a valid Flask application instance.
- Default configuration values are applied correctly.
- Configuration overrides are honoured.
- All stub endpoints respond with the expected HTTP status codes.
- Error handlers are wired up correctly.
"""

from __future__ import annotations

import pytest
from flask import Flask

from ai_image_audit import create_app
from ai_image_audit import __version__


@pytest.fixture()
def app() -> Flask:
    """Create a test Flask application with TESTING mode enabled."""
    return create_app({"TESTING": True, "DEBUG": False})


@pytest.fixture()
def client(app: Flask):
    """Return a test client for the Flask application."""
    return app.test_client()


# --------------------------------------------------------------------------- #
# Factory smoke tests                                                         #
# --------------------------------------------------------------------------- #

def test_create_app_returns_flask_instance(app: Flask) -> None:
    """create_app should return a Flask instance."""
    assert isinstance(app, Flask)


def test_testing_flag_is_set(app: Flask) -> None:
    """TESTING config key should be True when passed in overrides."""
    assert app.config["TESTING"] is True


def test_default_threshold(app: Flask) -> None:
    """AI_FLAG_THRESHOLD should default to 0.5."""
    plain_app = create_app({"TESTING": True})
    assert plain_app.config["AI_FLAG_THRESHOLD"] == pytest.approx(0.5)


def test_threshold_override() -> None:
    """AI_FLAG_THRESHOLD should be overridable via config dict."""
    custom_app = create_app({"TESTING": True, "AI_FLAG_THRESHOLD": 0.75})
    assert custom_app.config["AI_FLAG_THRESHOLD"] == pytest.approx(0.75)


def test_default_max_images(app: Flask) -> None:
    """MAX_IMAGES_PER_JOB should default to 200."""
    assert app.config["MAX_IMAGES_PER_JOB"] == 200


def test_default_suggestions_per_image(app: Flask) -> None:
    """SUGGESTIONS_PER_IMAGE should default to 5."""
    assert app.config["SUGGESTIONS_PER_IMAGE"] == 5


# --------------------------------------------------------------------------- #
# Health-check endpoint                                                       #
# --------------------------------------------------------------------------- #

def test_health_check_status_200(client) -> None:
    """GET /api/health should return HTTP 200."""
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_check_json_body(client) -> None:
    """GET /api/health should return JSON with status and version."""
    response = client.get("/api/health")
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["version"] == __version__


# --------------------------------------------------------------------------- #
# Stub endpoint responses                                                     #
# --------------------------------------------------------------------------- #

def test_scan_endpoint_returns_501(client) -> None:
    """POST /api/scan should return 501 Not Implemented in Phase 1."""
    response = client.post("/api/scan", json={"target": "/tmp"})
    assert response.status_code == 501


def test_scan_endpoint_returns_json(client) -> None:
    """POST /api/scan should return a JSON body."""
    response = client.post("/api/scan", json={"target": "/tmp"})
    data = response.get_json()
    assert data is not None
    assert "status" in data


def test_report_endpoint_returns_501(client) -> None:
    """GET /api/report/<job_id> should return 501 Not Implemented in Phase 1."""
    response = client.get("/api/report/test-job-123")
    assert response.status_code == 501


def test_report_endpoint_includes_job_id(client) -> None:
    """GET /api/report/<job_id> JSON body should echo back the job_id."""
    response = client.get("/api/report/my-job-abc")
    data = response.get_json()
    assert data["job_id"] == "my-job-abc"


def test_suggestions_endpoint_returns_501(client) -> None:
    """GET /api/suggestions should return 501 Not Implemented in Phase 1."""
    response = client.get("/api/suggestions?q=forest")
    assert response.status_code == 501


def test_suggestions_endpoint_returns_json(client) -> None:
    """GET /api/suggestions should return a JSON body with a suggestions key."""
    response = client.get("/api/suggestions?q=forest")
    data = response.get_json()
    assert "suggestions" in data


# --------------------------------------------------------------------------- #
# Error handlers                                                              #
# --------------------------------------------------------------------------- #

def test_404_handler_returns_json(client) -> None:
    """Requests to unknown routes should receive a JSON 404 response."""
    response = client.get("/this/does/not/exist")
    assert response.status_code == 404
    data = response.get_json()
    assert data["error"] == "not_found"


# --------------------------------------------------------------------------- #
# Package-level smoke test                                                    #
# --------------------------------------------------------------------------- #

def test_version_string_is_defined() -> None:
    """__version__ should be a non-empty string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0
