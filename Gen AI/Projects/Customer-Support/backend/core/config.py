from functools import lru_cache 
#stores the results of expensive function calls and returns the cached result when the same inputs occur again. LRU stands for Least Recently Used, which means that when the cache is full, the least recently used items will be discarded first. This is useful for optimizing performance in scenarios where certain function calls are made frequently with the same parameters.

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Customer Support app"
    app_version: str ="1.0.0"
    environment:str = "development"

    llm_provider: str = "ollama"
    llm_api_key: str = ""
    llm_model: str = "qwen3:1.7b"
    llm_base_url: str = "http://localhost:11434"

    request_timeout:float = 60.0

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False,
        extra = "ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


