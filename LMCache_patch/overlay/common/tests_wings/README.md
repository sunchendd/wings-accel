# tests_wings

Wings-specific LMCache tests live here.

Current coverage is organized by requirement family:

- `test_cold_start_manifest.py`
  Basic US-KV1 manifest roundtrip and remove/update behavior.
- `test_local_disk_lifecycle.py`
  US-KV2 local-disk lifecycle coverage for restore, usage accounting, and
  manifest validation boundaries.
- `test_qat_runtime.py`
  US-KV3 QAT runtime helpers, probing, and compression-ratio accounting.
- `test_full_sync_hooks.py`
  Additional non-US-KV hook coverage retained as an auxiliary capability.
