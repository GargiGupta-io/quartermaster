"""Settings read from the environment or a local .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    # Used when the main model is rate limited or down. It has its own free allowance.
    # Empty turns the backup off.
    groq_fallback_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    fx_base_url: str = "https://api.frankfurter.dev/v1"
    data_dir: Path = Path(".data")
    # When set, every API call except the page and health check needs this key in the
    # X-API-Key header. Left empty, the service is open to anyone, protected by the limits below.
    api_key: str = ""
    # Optional per-role approver keys, as "Role=key" pairs separated by semicolons, such as
    # "Department Head=k1;Finance Controller=k2". When set, a decision is accepted only with
    # the key for the role it is made as. Left empty, anyone may approve (the open demo).
    approver_keys: str = ""
    # Calls that use the AI (new requests and approval decisions). 0 turns a limit off.
    limit_per_visitor: int = 5
    limit_window_seconds: int = 600
    limit_per_day: int = 100
