"""Compatibility package for legacy LocalStore imports."""

__all__ = ["LocalStoreKVStore", "_sanitize_shm_name"]

from .localstore_connector import LocalStoreKVStore, _sanitize_shm_name
