"""AI Image Audit package.

This package provides a Flask-based web tool for auditing image assets
(from local directories or website URLs) to detect AI-generated content,
generate metadata reports, and suggest royalty-free human-made alternatives.

The public API exposes the Flask application factory ``create_app`` which
configures and returns a ready-to-use ``Flask`` application instance.

Example usage::

    from ai_image_audit import create_app

    app = create_app()
    app.run(debug=True)
"""

from ai_image_audit.app import create_app

__all__ = ["create_app"]
__version__ = "0.1.0"
