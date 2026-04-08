import importlib
from collections.abc import Callable

from vllm.logger import init_logger
from vsparse.kvstore import KVStoreBase

logger = init_logger(__name__)


class KVStoreFactory:
    _registry: dict[str, Callable[[], type[KVStoreBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[KVStoreBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(cls, connector_name: str, config: dict) -> KVStoreBase:
        if connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            raise ValueError(f"Unsupported connector type: {connector_name}")
        assert issubclass(connector_cls, KVStoreBase)
        logger.info("Creating connector with name: %s", connector_name)
        return connector_cls(config)


KVStoreFactory.register_connector(
    "LocalStoreKVStore",
    "vsparse.connectors.localstore_connector",
    "LocalStoreKVStore",
)
