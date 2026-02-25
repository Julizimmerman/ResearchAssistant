"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env file.

    All values can be overridden via environment variables or a .env file
    in the project root.
    """

    AZURE_OPENAI_API_KEY: str = Field(
        default="", description="Azure OpenAI API key"
    )
    AZURE_OPENAI_ENDPOINT: str = Field(
        default="", description="Azure OpenAI endpoint URL"
    )
    AZURE_OPENAI_API_VERSION: str = Field(
        default="2024-10-21", description="Azure OpenAI API version"
    )
    AZURE_OPENAI_DEPLOYMENT_CHEAP: str = Field(
        default="gpt-4o-mini",
        description="Azure deployment name for simple/cheap tasks",
    )
    AZURE_OPENAI_DEPLOYMENT_EXPENSIVE: str = Field(
        default="gpt-4o",
        description="Azure deployment name for complex/expensive tasks",
    )
    SQLITE_CACHE_PATH: str = Field(
        default=".llm_cache.db",
        description="File path for the SQLite LLM response cache",
    )
    MAX_SUBTOPICS: int = Field(
        default=7,
        ge=1,
        le=15,
        description="Maximum number of subtopics the Investigator should generate",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
