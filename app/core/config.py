import yaml
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

class RedisConfig(BaseModel):
    url: str

class PostgresConfig(BaseModel):
    url: str

class RoutingConfig(BaseModel):
    latency_weight: float
    accuracy_weight: float
    health_weight: float
    load_weight: float

class SLOConfig(BaseModel):
    p95_ms: float
    p99_ms: float
    error_rate: float

class BatchingConfig(BaseModel):
    enabled: bool
    max_batch_size: int
    max_wait_ms: int

class CanaryConfig(BaseModel):
    percentage: int

class RollbackConfig(BaseModel):
    latency_multiplier: float
    max_error_rate: float

class AppConfig(BaseModel):
    server: ServerConfig
    redis: RedisConfig
    postgres: PostgresConfig
    routing: RoutingConfig
    slo: SLOConfig
    batching: BatchingConfig
    canary: CanaryConfig
    rollback: RollbackConfig

class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"
    config_path: str = "config.yaml"
    app_config: Optional[AppConfig] = None

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"

settings = Settings()

def load_config() -> None:
    config_file = Path(settings.config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            yaml_config = yaml.safe_load(f)
            settings.app_config = AppConfig(**yaml_config)

load_config()
