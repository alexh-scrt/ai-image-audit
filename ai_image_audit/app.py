"""Flask application factory for the AI Image Audit web tool.

This module contains the ``create_app`` factory function that builds and
configures the Flask application instance, registers all API blueprints,
and sets up static file serving for the single-page frontend.

Routes registered here (stubs to be fully wired in Phase 5):

- ``GET  /``                     – Serve the SPA index page.
- ``POST /api/scan``             – Start a scan job for a directory or URL.
- ``GET  /api/report/<job_id>``  – Retrieve the full audit report for a job.
- ``GET  /api/suggestions``      – Fetch royalty-free alternative suggestions.
- ``GET  /api/health``           – Simple health-check endpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from flask import Flask, jsonify, send_from_directory

logger = logging.getLogger(__name__)


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Application factory that creates and configures the Flask app.

    Args:
        config: Optional dictionary of configuration overrides.  Keys map
                directly to Flask configuration variables (e.g.
                ``{"TESTING": True, "DEBUG": False}``).

    Returns:
        A fully configured :class:`flask.Flask` application instance.
    """
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
        static_url_path="/static",
    )

    # ------------------------------------------------------------------ #
    # Default configuration                                               #
    # ------------------------------------------------------------------ #
    app.config.setdefault("SECRET_KEY", os.environ.get("SECRET_KEY", "dev-secret-key"))
    app.config.setdefault("DEBUG", os.environ.get("FLASK_DEBUG", "false").lower() == "true")
    app.config.setdefault("TESTING", False)
    # Threshold above which an image is flagged as likely AI-generated (0–1).
    app.config.setdefault(
        "AI_FLAG_THRESHOLD",
        float(os.environ.get("AI_FLAG_THRESHOLD", "0.5")),
    )
    # Maximum number of images to scan per job to avoid runaway requests.
    app.config.setdefault(
        "MAX_IMAGES_PER_JOB",
        int(os.environ.get("MAX_IMAGES_PER_JOB", "200")),
    )
    # Number of royalty-free alternatives to return per flagged image.
    app.config.setdefault(
        "SUGGESTIONS_PER_IMAGE",
        int(os.environ.get("SUGGESTIONS_PER_IMAGE", "5")),
    )

    # Apply any caller-supplied overrides.
    if config:
        app.config.update(config)

    # ------------------------------------------------------------------ #
    # Logging                                                             #
    # ------------------------------------------------------------------ #
    _configure_logging(app)

    # ------------------------------------------------------------------ #
    # Route registration                                                  #
    # ------------------------------------------------------------------ #
    _register_routes(app)

    logger.info(
        "AI Image Audit application created (debug=%s, threshold=%.2f)",
        app.config["DEBUG"],
        app.config["AI_FLAG_THRESHOLD"],
    )
    return app


# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #

def _configure_logging(app: Flask) -> None:
    """Set up basic logging for the application.

    Args:
        app: The Flask application instance to configure logging for.
    """
    log_level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app.logger.setLevel(log_level)


def _register_routes(app: Flask) -> None:
    """Attach all URL rules to the Flask application.

    Args:
        app: The Flask application instance to register routes on.
    """

    # ------------------------------------------------------------------ #
    # Health check                                                        #
    # ------------------------------------------------------------------ #

    @app.get("/api/health")
    def health_check():
        """Return a simple JSON health-check response.

        Returns:
            JSON object with ``status`` and ``version`` fields.
        """
        from ai_image_audit import __version__

        return jsonify({"status": "ok", "version": __version__}), 200

    # ------------------------------------------------------------------ #
    # SPA index page                                                      #
    # ------------------------------------------------------------------ #

    @app.get("/")
    def index():
        """Serve the single-page application entry point.

        Returns:
            The ``index.html`` file from the ``static`` folder.
        """
        static_folder = app.static_folder or os.path.join(
            os.path.dirname(__file__), "static"
        )
        return send_from_directory(static_folder, "index.html")

    # ------------------------------------------------------------------ #
    # Scan endpoint (stub – fully implemented in Phase 5)                 #
    # ------------------------------------------------------------------ #

    @app.post("/api/scan")
    def scan():
        """Accept a scan job request and return a job identifier.

        Expected JSON body::

            {
                "target": "<local-directory-path or website-URL>",
                "threshold": 0.5   // optional, overrides app default
            }

        Returns:
            JSON object containing ``job_id`` and ``status``.
        """
        # Full implementation in Phase 5.
        return (
            jsonify(
                {
                    "job_id": None,
                    "status": "not_implemented",
                    "message": "Scan endpoint will be fully implemented in Phase 5.",
                }
            ),
            501,
        )

    # ------------------------------------------------------------------ #
    # Report endpoint (stub – fully implemented in Phase 5)              #
    # ------------------------------------------------------------------ #

    @app.get("/api/report/<job_id>")
    def report(job_id: str):
        """Return the audit report for the given job.

        Args:
            job_id: The unique identifier returned by the scan endpoint.

        Returns:
            JSON object containing the full audit report or an error.
        """
        # Full implementation in Phase 5.
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "status": "not_implemented",
                    "message": "Report endpoint will be fully implemented in Phase 5.",
                }
            ),
            501,
        )

    # ------------------------------------------------------------------ #
    # Suggestions endpoint (stub – fully implemented in Phase 5)         #
    # ------------------------------------------------------------------ #

    @app.get("/api/suggestions")
    def suggestions():
        """Return royalty-free image suggestions for a given query.

        Query parameters:
            ``q``: Search query derived from flagged image tags.
            ``per_page``: Number of results to return (default: 5).

        Returns:
            JSON object containing a list of image suggestion objects.
        """
        # Full implementation in Phase 5.
        return (
            jsonify(
                {
                    "suggestions": [],
                    "status": "not_implemented",
                    "message": "Suggestions endpoint will be fully implemented in Phase 5.",
                }
            ),
            501,
        )

    # ------------------------------------------------------------------ #
    # Generic error handlers                                              #
    # ------------------------------------------------------------------ #

    @app.errorhandler(400)
    def bad_request(error):
        """Return a JSON 400 Bad Request response."""
        return jsonify({"error": "bad_request", "message": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error):
        """Return a JSON 404 Not Found response."""
        return jsonify({"error": "not_found", "message": str(error)}), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Return a JSON 500 Internal Server Error response."""
        logger.exception("Unhandled internal error: %s", error)
        return (
            jsonify({"error": "internal_server_error", "message": "An unexpected error occurred."}),
            500,
        )


def main() -> None:
    """Entry point for running the development server directly.

    Reads ``FLASK_HOST``, ``FLASK_PORT``, and ``FLASK_DEBUG`` environment
    variables to configure the development server.
    """
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    app = create_app({"DEBUG": debug})
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
