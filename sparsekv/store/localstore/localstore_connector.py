"""Legacy import shim for LocalStoreKVStore.

Older code imports LocalStore via ``vsparse.store.localstore.localstore_connector``.
The canonical implementation now lives under ``vsparse.connectors``.
"""

__all__ = ["LocalStoreKVStore", "_sanitize_shm_name"]

from vsparse.connectors.localstore_connector import LocalStoreKVStore
from vsparse.shared_index import _sanitize_shm_name
