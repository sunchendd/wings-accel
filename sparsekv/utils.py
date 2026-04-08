from typing import Any, Dict

import yaml

from vllm.logger import init_logger

logger = init_logger(__name__)


class Config:
    def __init__(self, kv_transfer_config: Any):
        self.kv_transfer_config = kv_transfer_config
        self.config: Dict[str, Any] = {}
        self._load_config()

    @staticmethod
    def load_config_from_yaml(file_path: str) -> Dict[str, Any]:
        if not file_path:
            logger.warning("No sparse config file path provided.")
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                if not isinstance(config, dict):
                    logger.warning(
                        f"Config file {file_path} does not contain a dictionary. "
                        "Returning empty config."
                    )
                    return {}
                logger.info(f"Loaded sparse config from {file_path}")
                return config
        except FileNotFoundError:
            logger.error(f"sparse config file not found: {file_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config file {file_path}: {e}")
            return {}

    def _load_config(self) -> None:
        has_extra_config = (
            self.kv_transfer_config is not None
            and hasattr(self.kv_transfer_config, "kv_connector_extra_config")
            and self.kv_transfer_config.kv_connector_extra_config is not None
        )
        if not has_extra_config:
            self.config = self._get_default_config()
        else:
            extra_config = self.kv_transfer_config.kv_connector_extra_config
            if extra_config == {}:
                self.config = self._get_default_config()
            else:
                self.config = dict(extra_config)
                logger.info("Using kv_connector_extra_config from terminal input")

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        config = {
            "sparse_connectors": [
                {
                    "connector_name": "DramKVStore",
                    "connector_config": {},
                }
            ]
        }
        logger.warning(f"No sparse connector config provided, using default {config}")
        return config

    def get_config(self) -> Dict[str, Any]:
        return self.config