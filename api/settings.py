"""Server-owned credentials. Public deployment starts with generation disabled."""
from dataclasses import dataclass
from functools import lru_cache
import os
from pydantic import SecretStr
from tools.nebius_client import load_env_file


@dataclass(frozen=True)
class Settings:
    api_key: SecretStr
    generation_enabled: bool

    @property
    def can_generate(self):
        key = self.api_key.get_secret_value()
        return self.generation_enabled and bool(key) and key not in ("replace_me", "your_nebius_api_key_here")


@lru_cache(maxsize=1)
def get_settings():
    # Render receives secrets from its dashboard; local development can use .env.
    if os.environ.get("RENDER") != "true":
        load_env_file()
    return Settings(api_key=SecretStr(os.environ.get("NEBIUS_API_KEY", "").strip()),
                    generation_enabled=os.environ.get("GENERATION_ENABLED", "false").lower() == "true")
