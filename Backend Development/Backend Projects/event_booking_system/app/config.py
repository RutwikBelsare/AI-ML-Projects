"""
Centralized app configuration.

Everything the app needs from the environment is declared here as a typed
field. Pydantic Settings reads `.env` automatically and validates types,
so a missing or malformed value fails fast at startup instead of deep
inside a request handler.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    DATABASE_URL: str

    # JWT / auth
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Redis
    REDIS_URL: str

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str


@lru_cache
def get_settings() -> Settings:
    """
    Cached so `.env` is parsed once per process, not on every call.
    Import and call get_settings() anywhere you need config instead of
    instantiating Settings() directly.
    """
    return Settings()
