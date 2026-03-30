"""Flask application factory for the AI Image Audit web tool.

This module contains the ``create_app`` factory function that builds and
configures the Flask application instance, registers all API routes, and
integrates the scanner, classifier, report, and suggester modules into a
cohesive REST API.

Routes:

- ``GET  /``                     – Serve the SPA index page.
- ``POST /api/scan``             – Start a scan job and return a full audit report.
- ``GET  /api/report/<job_id>``  – Retrieve a previously generated audit report.
- ``GET  /api/suggestions``      – Fetch royalty-free alternative suggestions.
- ``GET  /api/health``           – Simple health-check endpoint.

Scan jobs are processed synchronously and stored in an in-memory
dictionary keyed by a UUID job identifier.  This is suitable for
development and small-scale use; a production deployment would replace
this with a persistent job store.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

logger = logging.getLogger(__name__)

# In-memory job store:  job_id -> report dict
_JOB_STORE: dict[str, dict[str, Any]] = {}


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
    # Optional path to a fine-tuned classifier checkpoint.
    app.config.setdefault(
        "CLASSIFIER_MODEL_PATH",
        os.environ.get("CLASSIFIER_MODEL_PATH", None),
    )
    # Torch device override (e.g. "cpu", "cuda").
    app.config.setdefault(
        "CLASSIFIER_DEVICE",
        os.environ.get("CLASSIFIER_DEVICE", "cpu"),
    )
    # Openverse API base URL (overridable for testing).
    app.config.setdefault(
        "OPENVERSE_API_BASE",
        os.environ.get("OPENVERSE_API_BASE", "https://api.openverse.org/v1"),
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
    # Scan endpoint                                                       #
    # ------------------------------------------------------------------ #

    @app.post("/api/scan")
    def scan():
        """Accept a scan job request, run the full audit pipeline, and store results.

        Expected JSON body::

            {
                "target": "<local-directory-path or website-URL>",
                "threshold": 0.5   // optional, overrides app default
            }

        The endpoint:

        1. Validates the request body.
        2. Invokes the scanner to collect image references.
        3. Runs the classifier on each image.
        4. Generates a metadata report per image.
        5. Stores the aggregated results under a new job UUID.
        6. Returns the job ID and a summary of the results.

        Returns:
            JSON object containing ``job_id``, ``status``, ``total``,
            ``flagged``, and ``images`` (list of serialised reports).

        HTTP status codes:
            200: Scan completed successfully.
            400: Invalid or missing request parameters.
            500: Internal processing error.
        """
        # --- Parse request body ---
        body = request.get_json(silent=True) or {}
        target = body.get("target", "") if isinstance(body, dict) else ""

        if not target or not str(target).strip():
            return (
                jsonify({
                    "error": "bad_request",
                    "message": "'target' field is required and must be a non-empty string.",
                }),
                400,
            )

        target = str(target).strip()

        # Allow per-request threshold override.
        try:
            threshold = float(
                body.get("threshold", app.config["AI_FLAG_THRESHOLD"])
            )
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("threshold out of range")
        except (TypeError, ValueError):
            return (
                jsonify({
                    "error": "bad_request",
                    "message": "'threshold' must be a float in [0.0, 1.0].",
                }),
                400,
            )

        max_images: int = app.config["MAX_IMAGES_PER_JOB"]
        model_path: str | None = app.config.get("CLASSIFIER_MODEL_PATH")
        device: str = app.config.get("CLASSIFIER_DEVICE", "cpu")

        # --- Import pipeline modules ---
        from ai_image_audit.scanner import scan as do_scan
        from ai_image_audit.classifier import AIImageClassifier
        from ai_image_audit.report import generate_report_from_path, generate_report

        # --- Discover images ---
        try:
            image_refs = do_scan(target, max_images=max_images)
        except FileNotFoundError as exc:
            return (
                jsonify({"error": "not_found", "message": str(exc)}),
                400,
            )
        except NotADirectoryError as exc:
            return (
                jsonify({"error": "bad_request", "message": str(exc)}),
                400,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("scan: error discovering images for target %r", target)
            return (
                jsonify({
                    "error": "scan_error",
                    "message": f"Failed to scan target: {exc}",
                }),
                500,
            )

        # --- Initialise classifier ---
        try:
            classifier = AIImageClassifier(
                model_path=model_path,
                threshold=threshold,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("scan: failed to initialise classifier")
            return (
                jsonify({
                    "error": "classifier_error",
                    "message": f"Failed to initialise classifier: {exc}",
                }),
                500,
            )

        # --- Process each image ---
        reports = []
        for ref in image_refs:
            try:
                if ref.is_remote:
                    # For remote images: download, classify, report.
                    report_dict = _process_remote_image(ref, classifier, threshold)
                else:
                    # For local images: classify via path, then generate report.
                    classification = classifier.classify_path(ref.source)
                    report = generate_report_from_path(ref, classification)
                    report_dict = report.to_dict()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "scan: error processing image %s – %s", ref.source, exc
                )
                report_dict = {
                    "source": ref.source,
                    "is_remote": ref.is_remote,
                    "origin": ref.origin,
                    "alt_text": ref.alt_text,
                    "extension": ref.extension,
                    "error": str(exc),
                    "ai_score": None,
                    "ai_is_flagged": None,
                    "ai_threshold": threshold,
                    "ai_verdict": None,
                    "ai_model_mode": None,
                    "width": None,
                    "height": None,
                    "aspect_ratio": None,
                    "file_size_bytes": None,
                    "format": None,
                    "mode": None,
                    "colour_stats": None,
                }
            reports.append(report_dict)

        # --- Aggregate results ---
        flagged = [
            r for r in reports if r.get("ai_is_flagged") is True
        ]
        job_id = str(uuid.uuid4())
        job_result = {
            "job_id": job_id,
            "status": "complete",
            "target": target,
            "threshold": threshold,
            "total": len(reports),
            "flagged_count": len(flagged),
            "images": reports,
        }
        _JOB_STORE[job_id] = job_result

        logger.info(
            "scan: job %s complete – %d image(s) scanned, %d flagged",
            job_id,
            len(reports),
            len(flagged),
        )
        return jsonify(job_result), 200

    # ------------------------------------------------------------------ #
    # Report endpoint                                                     #
    # ------------------------------------------------------------------ #

    @app.get("/api/report/<job_id>")
    def report(job_id: str):
        """Return the stored audit report for the given job ID.

        Args:
            job_id: The unique identifier returned by the ``/api/scan`` endpoint.

        Returns:
            JSON object containing the full audit report, or a 404 error if
            the job ID is not found.

        HTTP status codes:
            200: Report found and returned.
            404: No report found for the given job ID.
        """
        job_result = _JOB_STORE.get(job_id)
        if job_result is None:
            return (
                jsonify({
                    "error": "not_found",
                    "message": f"No report found for job_id {job_id!r}.",
                    "job_id": job_id,
                }),
                404,
            )
        return jsonify(job_result), 200

    # ------------------------------------------------------------------ #
    # Suggestions endpoint                                                #
    # ------------------------------------------------------------------ #

    @app.get("/api/suggestions")
    def suggestions():
        """Return royalty-free image suggestions for a given query.

        Query parameters:
            ``q``: Search query derived from flagged image tags or alt-text.
                   Required; returns 400 if missing or empty.
            ``per_page``: Number of results to return (default: value from
                   ``SUGGESTIONS_PER_IMAGE`` config, capped at 20).

        Returns:
            JSON object containing ``query``, ``per_page``, and
            ``suggestions`` (list of serialised :class:`ImageSuggestion`
            objects).

        HTTP status codes:
            200: Suggestions fetched successfully (list may be empty).
            400: Missing or invalid query parameter.
            502: Upstream Openverse API error.
        """
        query = request.args.get("q", "").strip()
        if not query:
            return (
                jsonify({
                    "error": "bad_request",
                    "message": "Query parameter 'q' is required and must be non-empty.",
                }),
                400,
            )

        default_per_page: int = app.config.get("SUGGESTIONS_PER_IMAGE", 5)
        try:
            per_page = int(request.args.get("per_page", default_per_page))
            if per_page < 1:
                raise ValueError("per_page must be >= 1")
        except (TypeError, ValueError):
            return (
                jsonify({
                    "error": "bad_request",
                    "message": "'per_page' must be a positive integer.",
                }),
                400,
            )

        api_base: str = app.config.get("OPENVERSE_API_BASE", "https://api.openverse.org/v1")

        from ai_image_audit.suggester import fetch_suggestions
        import requests as req_lib

        try:
            suggestion_list = fetch_suggestions(
                query,
                per_page=per_page,
                api_base=api_base,
            )
        except req_lib.exceptions.RequestException as exc:
            logger.error("suggestions: Openverse API error for query %r: %s", query, exc)
            return (
                jsonify({
                    "error": "upstream_error",
                    "message": f"Failed to fetch suggestions from Openverse: {exc}",
                    "suggestions": [],
                }),
                502,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("suggestions: unexpected error for query %r", query)
            return (
                jsonify({
                    "error": "internal_server_error",
                    "message": f"Unexpected error fetching suggestions: {exc}",
                    "suggestions": [],
                }),
                500,
            )

        return (
            jsonify({
                "query": query,
                "per_page": per_page,
                "total": len(suggestion_list),
                "suggestions": [s.to_dict() for s in suggestion_list],
            }),
            200,
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
            jsonify({
                "error": "internal_server_error",
                "message": "An unexpected error occurred.",
            }),
            500,
        )


# --------------------------------------------------------------------------- #
# Remote image processing helper                                              #
# --------------------------------------------------------------------------- #


def _process_remote_image(
    ref: Any,
    classifier: Any,
    threshold: float,
) -> dict[str, Any]:
    """Download a remote image, classify it, and generate a metadata report.

    Args:
        ref: An :class:`~ai_image_audit.scanner.ImageRef` with ``is_remote=True``.
        classifier: An initialised :class:`~ai_image_audit.classifier.AIImageClassifier`.
        threshold: The flagging threshold to include in the report.

    Returns:
        A JSON-compatible dictionary representing the
        :class:`~ai_image_audit.report.ImageReport`.

    Raises:
        requests.exceptions.RequestException: If the image cannot be downloaded.
        OSError: If the downloaded data cannot be opened as an image.
    """
    import io
    import requests as req_lib
    from PIL import Image, UnidentifiedImageError
    from ai_image_audit.report import generate_report
    from ai_image_audit.classifier import ClassificationResult

    _USER_AGENT = (
        "Mozilla/5.0 (compatible; AIImageAuditBot/0.1; "
        "+https://github.com/ai-image-audit)"
    )
    headers = {"User-Agent": _USER_AGENT}

    response = req_lib.get(ref.source, timeout=15, headers=headers)
    response.raise_for_status()

    try:
        img = Image.open(io.BytesIO(response.content))
        img.load()
    except UnidentifiedImageError as exc:
        raise OSError(
            f"Cannot identify remote image (corrupt or unsupported): {ref.source!r}"
        ) from exc

    classification: ClassificationResult = classifier.classify(img)

    # Build a minimal ref with file_size derived from the downloaded content.
    from ai_image_audit.report import ImageReport, ColourStats, compute_colour_stats

    width, height = img.size
    aspect_ratio = round(width / height, 4) if height > 0 else None
    file_size_bytes = len(response.content)
    img_format = img.format or "UNKNOWN"
    img_mode = img.mode

    try:
        colour_stats = compute_colour_stats(img)
        colour_stats_dict: dict[str, Any] | None = colour_stats.to_dict()
    except Exception:  # noqa: BLE001
        colour_stats_dict = None

    is_flagged = classification.is_flagged
    verdict = classification.verdict

    return {
        "source": ref.source,
        "is_remote": ref.is_remote,
        "origin": ref.origin,
        "alt_text": ref.alt_text,
        "extension": ref.extension,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "file_size_bytes": file_size_bytes,
        "format": img_format,
        "mode": img_mode,
        "colour_stats": colour_stats_dict,
        "ai_score": round(classification.score, 4),
        "ai_is_flagged": is_flagged,
        "ai_threshold": classification.threshold,
        "ai_verdict": verdict,
        "ai_model_mode": classification.model_mode,
        "error": None,
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


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
