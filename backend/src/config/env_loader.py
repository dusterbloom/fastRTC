"""Environment configuration loading with priority system."""

import os
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Configuration loader with environment priority."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.environment = os.getenv("ENVIRONMENT", "development")
        
    def load_environment(self) -> Dict[str, str]:
        """Load environment variables with priority system.
        
        Priority (highest to lowest):
        1. System environment variables
        2. .env.{environment} file
        3. .env file
        4. Default values
        """
        config = {}
        
        # Load base .env file
        env_file = self.base_path / ".env"
        if env_file.exists():
            config.update(self._load_env_file(env_file))
            logger.info(f"Loaded base configuration from {env_file}")
        
        # Load environment-specific .env file
        env_specific_file = self.base_path / f".env.{self.environment}"
        if env_specific_file.exists():
            config.update(self._load_env_file(env_specific_file))
            logger.info(f"Loaded {self.environment} configuration from {env_specific_file}")
        
        # System environment variables override everything
        system_env = {k: v for k, v in os.environ.items() if k.startswith(('FASTRTC_', 'OLLAMA_', 'LM_STUDIO_', 'AMEM_', 'QDRANT_', 'REDIS_'))}
        config.update(system_env)
        
        if system_env:
            logger.info(f"Applied {len(system_env)} system environment overrides")
        
        return config
    
    def _load_env_file(self, file_path: Path) -> Dict[str, str]:
        """Load environment variables from a .env file."""
        config = {}
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip().strip('"\'')
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
        return config