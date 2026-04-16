# qat

QAT codec helpers, runtime probing, and local-disk integration live here.

Vendored `kv-agent` source is kept outside the Wings-owned module code at:

`third_party/kv-agent/`

## Current integration status

- `build_wheel.sh` attempts an optional vendored `kv-agent` build before the
  LMCache wheel.
- `build_kv_agent.sh` checks for QATzip headers and runtime libraries before
  compiling.
- on `aarch64/arm64`, or when QAT prerequisites are missing, the optional
  common-path build skips `kv-agent` and the LMCache wheel still builds with
  raw local-disk fallback.
- runtime hooks can detect either a system-installed `kv_agent` or the vendored
  build output
- `LocalDiskBackend` save/load now routes through `wings_ext.qat.manager` when
  the probe succeeds
- if only vendored source exists but `_C` is not built yet, QAT stays disabled
- MLA is still unsupported in the current QAT runtime

## Runtime behavior

- feature gate: `extra_config.wings.qat.enabled`
- Python module override: `extra_config.wings.qat.module_name`
- vendored source override: `extra_config.wings.qat.kv_agent_root`
- device-count override for testing: `WINGS_QAT_AVAILABLE_DEVICES`

When any prerequisite is missing, the backend falls back to the original raw
disk path and records the reason in `backend.wings_qat_probe.reason`.
