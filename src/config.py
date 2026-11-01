"""Settings loaded from config.yaml, overridable by environment variables and .env.

Precedence (highest first): explicit arguments > ``MMRAG_*`` environment variables > ``.env`` >
``config.yaml`` > defaults below. Nested keys use ``__``, e.g. ``MMRAG_LLM__MODEL``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.yaml"


class QdrantSettings(BaseModel):
    path: str | None = "data/qdrant"
    url: str | None = None
    api_key: str | None = None
    collection: str = "mmrag_chunks"


class ExtractionSettings(BaseModel):
    output_dir: str = "data/processed/images"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    image_dpi: int = 150
    table_backend: Literal["pymupdf", "unstructured"] = "pymupdf"


class TextEmbeddingSettings(BaseModel):
    provider: Literal["litellm", "local"] = "litellm"
    model: str = "gemini/gemini-embedding-001"
    dimension: int | None = 3072
    api_base: str | None = None

    @model_validator(mode="after")
    def _dimension_for_api(self) -> TextEmbeddingSettings:
        if self.provider == "litellm" and not self.dimension:
            raise ValueError("embedding.text.dimension is required when provider is 'litellm'")
        return self


class EmbeddingSettings(BaseModel):
    text: TextEmbeddingSettings = Field(default_factory=TextEmbeddingSettings)
    image_strategy: Literal["caption", "clip", "both"] = "caption"
    caption_model: str = "gemini/gemini-3.8-flash"
    caption_concurrency: int = 2
    caption_api_base: str | None = None
    clip_model: str = "sentence-transformers/clip-ViT-B-32"


class RetrievalSettings(BaseModel):
    top_k: int = 6
    max_images: int = 3
    max_context_chars: int = 12_000
    reranker: Literal["none", "keyword", "cross-encoder"] = "keyword"
    reranker_weight: float = 0.5
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LLMSettings(BaseModel):
    model: str = "gemini/gemini-3.8-flash"
    fallback_models: list[str] = Field(default_factory=list)
    temperature: float = 0.1
    max_tokens: int = 1024
    api_base: str | None = None
    send_images: bool | None = None


class Settings(BaseSettings):
    """All pipeline settings. Build with ``load_settings()``."""

    model_config = SettingsConfigDict(
        env_prefix="MMRAG_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    log_level: str = "INFO"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls))


def load_settings(
    config_path: str | Path | None = None, env_file: str | Path | None = ".env", **overrides: Any
) -> Settings:
    """Load settings from YAML + environment.

    Also loads ``env_file`` into ``os.environ`` (without overwriting existing variables) so that
    litellm can find provider API keys such as ``GEMINI_API_KEY``.

    Args:
        config_path: YAML file; defaults to ``$MMRAG_CONFIG`` or ``config.yaml``. A missing default
            file is fine (built-in defaults apply); a missing explicit file raises.
        env_file: ``.env`` file to load, or ``None`` to skip.
        **overrides: Top-level fields to set explicitly (highest precedence).

    Raises:
        FileNotFoundError: If an explicitly given ``config_path`` does not exist.
    """
    explicit = config_path is not None or "MMRAG_CONFIG" in os.environ
    path = Path(config_path or os.environ.get("MMRAG_CONFIG", DEFAULT_CONFIG_PATH))
    if explicit and not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    if env_file is not None and Path(env_file).is_file():
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)

    yaml_file = path if path.is_file() else None

    class _Settings(Settings):
        model_config = SettingsConfigDict(
            **{**Settings.model_config, "yaml_file": yaml_file, "env_file": env_file}
        )

    settings = _Settings(**overrides)
    logger.debug("Loaded settings from %s", yaml_file or "defaults")
    return Settings.model_validate(settings.model_dump())


def missing_api_keys(models: list[str]) -> dict[str, list[str]]:
    """Map each model to the environment variables litellm says it needs but cannot find."""
    import litellm

    missing: dict[str, list[str]] = {}
    for model in dict.fromkeys(models):
        try:
            report = litellm.validate_environment(model=model)
        except Exception:  # unknown provider; let the real call report the problem
            logger.debug("Could not check environment for %s", model, exc_info=True)
            continue
        keys = [str(k) for k in report.get("missing_keys", [])]
        if keys:
            missing[model] = keys
    return missing
