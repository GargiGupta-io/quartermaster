"""Settings read from the environment or a local .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    fx_base_url: str = "https://api.frankfurter.dev/v1"
    data_dir: Path = Path(".data")
    # When set, every API call except the page and health check needs this key in the
    # X-API-Key header. Left empty for local use only.
    api_key: str = ""
