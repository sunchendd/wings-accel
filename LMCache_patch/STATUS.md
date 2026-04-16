# LMCache Patch Status

This file tracks the Wings LMCache patch-first integration inside
`LMCache_patch/`, aligned to the current `US-KV1 ~ US-KV4` requirement set.

## Requirement Mapping

- `P0 / US-KV1`: Cold Start
- `P1 / US-KV2`: Local-Disk Lifecycle Management
- `P2 / US-KV3`: QAT Compression, including ARM degradation
- `P3 / US-KV4`: Ascend/NPU support, reusing `US-KV1 + US-KV2` and explicitly not supporting `US-KV3`

## Status Legend

- `Implemented`: usable in the current patch-first workflow
- `Mostly Implemented`: main path is present, but the requirement still needs sharper validation or requirement-specific coverage
- `Partial`: important parts exist, but there are still material gaps
- `Not Migrated`: not present in the current implementation

## P0 / US-KV1 Cold Start

Overall status: `Implemented`

| Capability | Status | Current implementation | Gap / note |
|---|---|---|---|
| Manifest-based cold restore | Implemented | [`0002-local-disk-hooks.patch`](/workspace/wings-accel/LMCache_patch/patches/common/v0.3.15/0002-local-disk-hooks.patch) initializes cold-start state in `LocalDiskBackend`, and [`cold_start/hooks.py`](/workspace/wings-accel/LMCache_patch/overlay/common/lmcache/v1/wings_ext/cold_start/hooks.py) restores entries from the manifest | This is a manifest design, not the old file-scan design |
| Incremental manifest updates on write/remove | Implemented | [`cold_start/hooks.py`](/workspace/wings-accel/LMCache_patch/overlay/common/lmcache/v1/wings_ext/cold_start/hooks.py) updates snapshot state on write/remove | No background reconciliation loop |
| Close-time flush | Implemented | [`0002-local-disk-hooks.patch`](/workspace/wings-accel/LMCache_patch/patches/common/v0.3.15/0002-local-disk-hooks.patch) forces `save_manifest(..., force=True)` on backend close | None |
| Restore-time safety checks | Implemented | Missing payloads, invalid manifests, and storage-path mismatches are skipped in [`cold_start/hooks.py`](/workspace/wings-accel/LMCache_patch/overlay/common/lmcache/v1/wings_ext/cold_start/hooks.py) | No salvage path for broken manifests |
| Cold-start tests | Implemented | [`test_cold_start_manifest.py`](/workspace/wings-accel/LMCache_patch/overlay/common/tests_wings/test_cold_start_manifest.py) covers roundtrip and remove/update behavior | No end-to-end runtime restart smoke yet |
| Old file-scan recovery model | Not Migrated | The current implementation intentionally replaced the old scan-and-infer approach | Design change, not an accidental gap |

## P1 / US-KV2 Local-Disk Lifecycle Management

Overall status: `Mostly Implemented`

Assumed requirement boundary: `LocalDiskBackend` init / put / get / remove / evict / close / usage accounting.

| Capability | Status | Current implementation | Gap / note |
|---|---|---|---|
| Init-time state restore and usage rebuild | Implemented | [`0002-local-disk-hooks.patch`](/workspace/wings-accel/LMCache_patch/patches/common/v0.3.15/0002-local-disk-hooks.patch) restores cold-start state during backend init and rebuilds `current_cache_size` / `usage` through [`cold_start/hooks.py`](/workspace/wings-accel/LMCache_patch/overlay/common/lmcache/v1/wings_ext/cold_start/hooks.py) | None |
| Put-path metadata persistence | Implemented | Backend save path updates manifest state after `insert_key(...)` through [`0002-local-disk-hooks.patch`](/workspace/wings-accel/LMCache_patch/patches/common/v0.3.15/0002-local-disk-hooks.patch) | No dedicated end-to-end put/get test against upstream backend class |
| Remove-path lifecycle persistence | Implemented | Backend remove path calls `note_manifest_remove(...)` through [`0002-local-disk-hooks.patch`](/workspace/wings-accel/LMCache_patch/patches/common/v0.3.15/0002-local-disk-hooks.patch) | None |
| Close-path flush | Implemented | Backend close path flushes manifest through [`0002-local-disk-hooks.patch`](/workspace/wings-accel/LMCache_patch/patches/common/v0.3.15/0002-local-disk-hooks.patch) | None |
| Usage accounting after restore | Implemented | [`cold_start/hooks.py`](/workspace/wings-accel/LMCache_patch/overlay/common/lmcache/v1/wings_ext/cold_start/hooks.py) restores `current_cache_size` and `usage` from rebuilt disk metadata | No explicit upstream backend smoke test |
| Requirement-specific lifecycle tests | Implemented | [`test_local_disk_lifecycle.py`](/workspace/wings-accel/LMCache_patch/overlay/common/tests_wings/test_local_disk_lifecycle.py) now covers restore-time usage rebuild, missing-payload skip, and storage-path mismatch rejection | Still lighter than a real `LocalDiskBackend` integration test |
| Eviction-path requirement coverage | Partial | Eviction still flows through upstream backend logic, and size accounting is wired via the same local-disk hook patch | No dedicated eviction-specific test yet |

## P2 / US-KV3 QAT Compression

Overall status: `Partial`

| Capability | Status | Current implementation | Gap / note |
|---|---|---|---|
| Vendored `kv-agent` source in repo | Implemented | [`overlay/common/third_party/kv-agent`](/workspace/wings-accel/LMCache_patch/overlay/common/third_party/kv-agent) is part of the patch-first tree | None |
| Standalone QAT build entrypoint | Implemented | [`install.py`](/workspace/wings-accel/LMCache_patch/install.py) exposes `build-kv-agent` | None |
| QAT runtime module detection | Implemented | [`qat/hooks.py`](/workspace/wings-accel/LMCache_patch/overlay/common/lmcache/v1/wings_ext/qat/hooks.py) supports system-installed or vendored `kv_agent` | None |
| QAT device probing | Implemented | [`qat/device_probe.py`](/workspace/wings-accel/LMCache_patch/overlay/common/lmcache/v1/wings_ext/qat/device_probe.py) supports `adf_ctl` first and `lspci -nn` fallback with node grouping | No standalone validator/reporting tool |
| Local-disk QAT save/load path | Implemented | [`0002-local-disk-hooks.patch`](/workspace/wings-accel/LMCache_patch/patches/common/v0.3.15/0002-local-disk-hooks.patch) routes save/load through Wings QAT hooks | Falls back to raw disk I/O when QAT is unavailable |
| Compression-ratio tracking and capacity correction | Implemented | [`qat/manager.py`](/workspace/wings-accel/LMCache_patch/overlay/common/lmcache/v1/wings_ext/qat/manager.py) and [`qat/hooks.py`](/workspace/wings-accel/LMCache_patch/overlay/common/lmcache/v1/wings_ext/qat/hooks.py) reserve estimated size and reconcile using the real persisted file size | Ratio is process-local and local-disk-scoped |
| ARM degradation for common wheel builds | Implemented | [`build_kv_agent.sh`](/workspace/wings-accel/LMCache_patch/scripts/build_kv_agent.sh) now treats QAT as optional in build-wheel flow, skips `kv-agent` on `aarch64/arm64`, and allows LMCache wheel build to continue | Direct `build-kv-agent` remains strict unless `--optional` is used |
| Degradation on missing QAT toolchain or runtime libs | Implemented | Optional `kv-agent` builds now skip cleanly when headers, libs, or toolchain are missing, and [`build_wheel.sh`](/workspace/wings-accel/LMCache_patch/scripts/build_wheel.sh) continues building the LMCache wheel | QAT wheel is not produced in this mode |
| QAT helper tests | Implemented | [`test_qat_runtime.py`](/workspace/wings-accel/LMCache_patch/overlay/common/tests_wings/test_qat_runtime.py) covers manager/hook save-load behavior, probing, and compression-ratio accounting | No real hardware-backed QAT smoke test yet |
| MLA support | Not Migrated | Current QAT runtime rejects MLA explicitly | Needs additional format/runtime work |
| Full old-Wings validator tooling | Not Migrated | The old validator has not been ported | Still missing if parity is required |

## P3 / US-KV4 Ascend / NPU Support

Overall status: `Partial`

Required boundary: reuse `US-KV1 + US-KV2`, do not support `US-KV3`.

| Capability | Status | Current implementation | Gap / note |
|---|---|---|---|
| `common + ascend` layering | Implemented | [`materialize_workspace.sh`](/workspace/wings-accel/LMCache_patch/scripts/materialize_workspace.sh) and [`apply_patchset.sh`](/workspace/wings-accel/LMCache_patch/scripts/apply_patchset.sh) apply common first, then Ascend-specific overlay/patches | None |
| Offline/internal `kvcache-ops` source workflow | Implemented | `kvcache-ops` is locked under `third_party_sources/`, and [`prepare_ascend_sources.sh`](/workspace/wings-accel/LMCache_patch/scripts/prepare_ascend_sources.sh) handles the internal Git preprocessing step | Internal Git path cannot be validated in this external environment |
| Runtime NPU detection and device routing | Implemented | [`runtime.py`](/workspace/wings-accel/LMCache_patch/overlay/ascend/lmcache/v1/wings_ext/ascend/runtime.py) detects `npu` and drives backend dst-device selection | Some vLLM-Ascend edge paths may still need follow-up |
| Ascend connector first pass | Partial | [`npu_connectors.py`](/workspace/wings-accel/LMCache_patch/overlay/ascend/lmcache/v1/gpu_connector/npu_connectors.py) supports non-layerwise NPU layouts with native-first copy and Python fallback | Layerwise/blending and broader layout coverage are still missing |
| Ascend build-mode hook | Implemented | [`0003-ascend-build-hooks.patch`](/workspace/wings-accel/LMCache_patch/patches/ascend/v0.3.15/0003-ascend-build-hooks.patch) builds `kvcache-ops`, compiles Ascend `c_ops`, and bundles runtime artifacts | Needs a real internal Ascend build validation |
| Explicit no-QAT rule | Implemented | [`0002-ascend-qat-unsupported.patch`](/workspace/wings-accel/LMCache_patch/patches/ascend/v0.3.15/0002-ascend-qat-unsupported.patch) makes QAT fail fast on NPU | Matches the current requirement |
| Ascend tests | Partial | [`test_ascend_runtime.py`](/workspace/wings-accel/LMCache_patch/overlay/ascend/tests_wings/test_ascend_runtime.py) covers runtime helper behavior | No hardware-backed save/load smoke test yet |

## Additional Capabilities Outside US-KV1 ~ US-KV4 Mainline

These capabilities are currently present, but they are not part of the main requirement chain above:

- `maintenance`
  Process-local maintenance mode state, internal API wiring, and vLLM gating are present.
- `full_sync`
  Hook-based lifecycle interception, cooldown policy, and bounded history tracking are present.

They should be treated as auxiliary capabilities rather than the main delivery line for the current milestone.

## Recommended Next Steps

1. Run one real common-path wheel build on an ARM or QAT-free machine to confirm the new optional QAT degradation behavior end to end.
2. Add an eviction-focused `US-KV2` test so the disk lifecycle requirement has explicit coverage for eviction pressure.
3. Run one real x86 + QAT validation pass to prove save/load and compressed-size reconciliation on hardware.
4. Run one real Ascend validation pass covering `import lmcache`, NPU detection, cold restore, local-disk save/load, and explicit QAT rejection.
