"""Deployment and environment configuration."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    enable_file_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true")
    log_file_path: str = field(default_factory=lambda: os.getenv("LOG_FILE_PATH", "./logs/voice_assistant.log"))
    max_file_size: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_FILE_SIZE", "10485760")))  # 10MB
    backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "5")))
    
    # Structured logging
    enable_json_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_JSON_LOGGING", "false").lower() == "true")

@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_cors: bool = field(default_factory=lambda: os.getenv("ENABLE_CORS", "true").lower() == "true")
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    
    # API security
    enable_api_key: bool = field(default_factory=lambda: os.getenv("ENABLE_API_KEY", "false").lower() == "true")
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("API_KEY"))
    
    # Rate limiting
    enable_rate_limiting: bool = field(default_factory=lambda: os.getenv("ENABLE_RATE_LIMITING", "false").lower() == "true")
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "100")))
    rate_limit_window: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW", "3600")))  # 1 hour

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = field(default_factory=lambda: os.getenv("ENABLE_METRICS", "false").lower() == "true")
    metrics_port: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "9090")))
    
    enable_tracing: bool = field(default_factory=lambda: os.getenv("ENABLE_TRACING", "false").lower() == "true")
    jaeger_endpoint: Optional[str] = field(default_factory=lambda: os.getenv("JAEGER_ENDPOINT"))
    
    enable_health_checks: bool = field(default_factory=lambda: os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true")
    health_check_interval: float = field(default_factory=lambda: float(os.getenv("HEALTH_CHECK_INTERVAL", "30.0")))