"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from protondb_settings.config import Config
from protondb_settings.api.routes.health import router as health_router


def create_app(config: Config | None = None) -> FastAPI:
    """Build and return a configured FastAPI application."""
    if config is None:
        config = Config()

    app = FastAPI(
        title="ProtonDB Recommended Settings",
        version="0.1.0",
        description="API for optimal Linux gaming settings based on ProtonDB reports.",
    )

    # Store config for dependency injection in routes.
    app.state.config = config

    app.include_router(health_router)

    return app
