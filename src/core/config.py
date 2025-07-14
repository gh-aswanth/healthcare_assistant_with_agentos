import os
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        validate_default=False,
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    CLI_BACKEND_ORIGIN_URL: str = Field(default="http://localhost:8000")
    USERNAME: str = Field(default="admin")
    PASSWORD: str = Field(default="Dev@aiml2025")
    WORKER_COUNT: int = Field(default=1)
    CONCURRENT_THREAD_COUNT: int = Field(default=100)
    OPENAI_API_KEY: str = Field(default="")
    QDRANT_COLLECTION: str = Field(default="vector_collection")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-large")
    OPENAI_EMBEDDING_DIMENSIONS: int = Field(default=3072)
    QDRANT_URL: str = Field(default="http://localhost:6333")
    EDGES_PER_NODE: int = 8
    CUSTOM_PAYLOAD_M: int = 16
    NEIGHBORS_NUM: int = 50
    MAX_INDEX_THREAD: int = 8
    DEFAULT_SEGMENT_NUM: int = 5
    MAX_RELEVANT_MATCHES: int = 10
    OVERSAMPLING_FACTOR: float = 3.0

    @model_validator(mode="after")
    def check_worker_config(self) -> Self:
        """

        :return:
        """
        os.environ.setdefault("OPENAI_API_KEY", self.OPENAI_API_KEY)
        return self

    @field_validator("CLI_BACKEND_ORIGIN_URL")
    def check_for_trailing_slash(cls, v: str):
        if v.endswith("/"):
            raise ValueError("'CLI_BACKEND_ORIGIN_URL' value must not have the trailing slash in the URL.")
        return v


@lru_cache
def get_settings():
    return Settings()
