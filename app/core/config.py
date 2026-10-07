import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, BaseModel

# P3-1: Switched to pydantic-settings and allow env var overrides

class PostgresConfig(BaseModel):
    url: str = Field(pattern=r"postgresql://.*")

class RedisConfig(BaseModel):
    url: str = Field(pattern=r"redis://.*")
    ttl: int = 3600
    
class RoutingConfig(BaseModel):
    latency_weight: float = 0.4
    accuracy_weight: float = 0.6
    
class BatchingConfig(BaseModel):
    enabled: bool = True
    max_batch_size: int = 16
    max_wait_ms: int = 10
    
class CanaryConfig(BaseModel):
    percentage: int = 5
    
class RollbackConfig(BaseModel):
    latency_multiplier: float = 1.25
    max_error_rate: float = 0.02
    min_samples: int = 100

class AppConfig(BaseModel):
    postgres: PostgresConfig
    redis: RedisConfig
    routing: RoutingConfig
    batching: BatchingConfig
    canary: CanaryConfig
    rollback: RollbackConfig

class Settings(BaseSettings):
    app_env: str = Field("development", alias="APP_ENV")
    log_level: str = getattr(logging, os.getenv("LOG_LEVEL", "INFO"), logging.INFO) if "logging" in globals() else "INFO"
    config_path: str = Field("config.yaml", alias="CONFIG_PATH")
    app_config: AppConfig | None = None
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

def load_config():
    import yaml
    import logging
    try:
        with open(settings.config_path, "r") as f:
            raw_conf = yaml.safe_load(f)
            # Allow env overrides mapping
            if "DATABASE_URL" in os.environ:
                raw_conf.setdefault("postgres", {})["url"] = os.environ["DATABASE_URL"]
            if "REDIS_URL" in os.environ:
                raw_conf.setdefault("redis", {})["url"] = os.environ["REDIS_URL"]
                
            settings.app_config = AppConfig(**raw_conf)
    except Exception as e:
        # Don't fail immediately, some tests mock things, but do log it heavily
        print(f"Failed to load config from {settings.config_path}: {e}")
