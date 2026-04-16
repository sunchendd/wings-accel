# LMCache Patch-First Workflow

This directory is the standalone integration workspace for LMCache inside
`wings-accel`. It keeps upstream LMCache as a clean tarball input, layers
Wings-owned code via `overlay/`, applies small hook patches from `patches/`,
and builds from a disposable generated workspace.

Current migration status is tracked against the `US-KV1 ~ US-KV4` requirement
set in [STATUS.md](/workspace/wings-accel/LMCache_patch/STATUS.md).

## Layout

- `manifest/`
  Source locks and reproducibility metadata for LMCache and extra sources.
- `upstream_sources/`
  Prepared LMCache source tarballs generated from the internal SSH repo.
- `third_party_sources/`
  Additional prepared source snapshots such as `QATzip` and `kvcache-ops`.
- `overlay/common/`
  Shared Wings-owned logic copied into every generated workspace.
- `overlay/ascend/`
  Ascend-only overlay additions.
- `patches/common/<version>/`
  Shared hook patches applied after overlay materialization.
- `patches/ascend/<version>/`
  Ascend-only hook patches.
- `scripts/`
  Reproducible build and source-preparation helpers.
- `build/generated/LMCache/`
  Disposable unpacked workspace created during replay.
- `dist/`
  Standalone wheel output directory.

## Golden Rules

- Keep upstream LMCache tarballs pristine.
- Keep `kvcache-ops` as a locked source snapshot, not a live Git worktree.
- Put Wings-owned feature logic in `overlay/common` or `overlay/ascend`.
- Keep upstream file edits thin and save them as small patches.
- Never treat `build/generated/LMCache/` as the long-term source of truth.
- Use `LMCache_patch/install.py` as the only convenience entrypoint for this subtree.

## Common Workflow

1. Prepare common-path sources from the pinned artifact URLs in `manifest/lmcache.lock.json`:

```bash
cd LMCache_patch
python3 install.py prepare-common-sources
```

2. Keep [manifest/lmcache.lock.json](/workspace/wings-accel/LMCache_patch/manifest/lmcache.lock.json) pinned to the required tarball URLs, names, root directories, and hashes.
3. Run:

```bash
cd LMCache_patch
python3 install.py verify-upgrade
python3 install.py build-wheel
```

`prepare-common-sources` only uses the following pinned artifacts:

- `https://artifactrepo.wux-g.tools.xfusion.com/artifactory/opensource_general/LMCache/v0.3.15/package/LMCache-0.3.15.tar.gz`
- `https://artifactrepo.wux-g.tools.xfusion.com/artifactory/opensource_general/QAT/v1.3.2/package/QATzip-1.3.2.tar.gz`

`build-wheel` for `platform=common` attempts an optional vendored `kv-agent`
build first and then builds the LMCache wheel. If the host is ARM, or if the
QAT toolchain/runtime prerequisites are missing, the `kv-agent` build is
skipped and the common LMCache wheel still builds with raw local-disk fallback.
Run `python3 install.py build-kv-agent` only when you want to debug the QAT
extension by itself in strict mode.

For the NVIDIA common path, LMCache also compiles its own CUDA `c_ops`
extension. That step requires CUDA development headers such as `cusparse.h`.
If the builder image only provides CUDA runtime libraries, use a CUDA devel
image or mount a host toolkit root through `WINGS_LMCACHE_CUDA_HOME` in the
top-level `build/build_lmcache.sh` flow. Missing `cusparse.h` is unrelated to
QATzip.

## Ascend Workflow

Ascend stays on the same ordinary LMCache package line. This tree does not
introduce `LMCache-Ascend` as a separate wheel or source fork.

Local development:

- `kvcache-ops` no longer stays vendored in this repo. Ascend builds fetch it from `ssh://git@git.codehub.xfusion.com:2222/OpenSourceCenter/kvcache-ops.git` and materialize it as `third_party/kvcache-ops` inside the generated workspace.

Internal company build:

- Run the explicit pre-processing step first:

```bash
cd LMCache_patch
python3 install.py prepare-ascend-sources
python3 install.py verify-upgrade --platform ascend
python3 install.py build-wheel --platform ascend
```

`prepare-ascend-sources` clones the locked `kvcache-ops` ref over SSH, archives
it under `third_party_sources/kvcache-ops/kvcache-ops.tar.gz`, and later
materializes it into the generated LMCache workspace as `third_party/kvcache-ops`.
If the container or host cannot access the internal Git server over SSH,
`verify-upgrade --platform ascend` and `build-wheel --platform ascend` fail fast.

Ascend currently keeps the old Wings rule that QAT is unsupported. The build
workflow therefore skips `kv-agent`, and `build-kv-agent --platform ascend`
fails intentionally.

## Hook Patch Layers

`patches/common/v0.3.15/` contains the current real hook-based patch set:

- `0001-config-hooks.patch`
  Adds `get_wings_config`, `get_wings_feature_config`, and
  `is_wings_feature_enabled` helpers to `LMCacheEngineConfig`.
- `0002-local-disk-hooks.patch`
  Adds local-disk hooks for Wings cold-start manifest persistence, restart
  restore, plus QAT save/load handoff, estimated compressed-size reservation,
  and persisted-size reconciliation back into the local-disk backend.
- `0003-internal-api-hooks.patch`
  Makes the internal FastAPI app instance-scoped so Wings APIs can be mounted
  safely per manager instance.
- `0004-full-sync-hooks.patch`
  Adds lifecycle hooks around full-sync start and finish, including
  cooldown-based skip decisions and bounded per-sender history tracking.
- `0005-vllm-adapter-hooks.patch`
  Adds maintenance-mode gating on vLLM lookup and save paths.

`patches/ascend/v0.3.15/` now contains the first Ascend-only hook set:

- `0001-ascend-runtime-hooks.patch`
  Enables NPU device detection, routes backend dst-device selection through the
  Ascend runtime helper, wires the NPU connector, and adds NUMA fallback logic.
- `0002-ascend-qat-unsupported.patch`
  Placeholder only. Ascend QAT rejection now lives in the shared
  `overlay/common/lmcache/v1/wings_ext/qat/hooks.py` path so the check stays in
  sync with the common QAT hook implementation.
- `0003-ascend-build-hooks.patch`
  Placeholder only. Ascend `setup.py` customization now lives in
  `overlay/ascend/setup.py`, which replaces upstream `setup.py` in the
  materialized workspace and avoids keeping a large brittle patch in sync.

## Current Feature Boundary

The shared `common` layer is aligned to the current main requirement chain:

- `US-KV1` cold start:
  manifest-based persistence, close-time flush, and restart restore are wired.
- `US-KV2` local-disk lifecycle:
  init/put/remove/close lifecycle hooks and usage/accounting rebuild are wired
  through the same local-disk patch path.
- `US-KV3` QAT:
  `kv-agent` is vendored, buildable, reconnected to local-disk save/load on
  the common path, and now updates local-disk capacity using tracked
  compression ratios and persisted file sizes.

Additional non-mainline capabilities still present in this tree:

- maintenance mode
- full-sync hook policy

The Ascend layer currently targets `US-KV4`:

- platform-aware overlay and patch application
- offline or internal-Git `kvcache-ops` source handling
- `third_party/kvcache-ops` materialization into the generated workspace
- explicit `platform=ascend` build/verify entrypoints
- `torch_npu.contrib.transfer_to_npu` compatibility shim
- `npu` runtime detection for vLLM
- non-layerwise NPU connector with native-first save/load for both
  `[2, nb, bs, nh, hs]` and `[nb, 2, bs, nh, hs]`, plus Python fallback
- `setup.py`-driven `kvcache-ops` build and wheel bundling for Ascend
- Ascend-specific `setup.py` is provided through overlay replacement instead of
  a long patch series
- backend dst-device selection updated for `npu`
- explicit no-QAT behavior for Ascend so `US-KV4` reuses `US-KV1 + US-KV2`
  without `US-KV3`

Still intentionally deferred for a later pass:

- layerwise/blending connector support on Ascend
- native handling for additional KV layouts beyond the current non-layerwise
  vLLM layouts
- broader runtime cleanup for special paths such as MLA/save-only-first-rank
- old-style QAT validator/reporting tooling

## QAT Build Notes

`scripts/build_kv_agent.sh` performs a dependency preflight before compiling.
The host must provide:

- `qatzip.h`
- `libqatzip.so`
- `libqat_s.so`
- `libusdm_drv_s.so`
- `libnuma.so`

If the libraries or headers live outside standard system directories, set
`QATZIP_INCLUDE_DIR` and/or `QATZIP_LIB_DIR` before running the build.

QATzip now follows the same pinned-artifact rule as the main LMCache tarball:

1. Run `python3 install.py prepare-common-sources` to download both pinned tarballs.
2. Keep `manifest/lmcache.lock.json` for `extra_sources.qatzip` aligned with the required URL, `tarball_name`, `tarball_sha256`, and `root_dir_name_in_tar`.
3. Provide a matching QAT package tree explicitly through `QAT_PACKAGE_ROOT`
   if the environment does not already expose `qatzip.h` and `libqatzip.so`
   in standard include/library paths.

Example:

```bash
export QAT_PACKAGE_ROOT=/path/to/offline/QAT
python3 install.py build-kv-agent
```

If `qatzip.h` and `libqatzip.so` are already installed, `build_kv_agent.sh`
uses them directly. Otherwise it extracts the locked tarball from
`third_party_sources/qatzip/`, builds QATzip locally against the explicit
`QAT_PACKAGE_ROOT`, and then stages the resulting headers and shared
libraries under the build workspace automatically.

The build no longer auto-discovers live source trees such as `../QATzip` or
`/harbor_data/wgd/QATzip`. Real build environments consume the pinned
`QATzip-1.3.2.tar.gz` snapshot plus explicit QAT runtime inputs.

If your environment already provides `qatzip.h`, `libqatzip.so`, `libqat_s.so`,
`libusdm_drv_s.so`, and `libnuma.so` in standard library/include locations, the
staging step is unnecessary.

When validating QAT inside a container, the shared libraries alone are not
enough. The container must also see the relevant UIO character devices exposed
by the host QAT driver. A practical rule is:

- prefer fixing host-side device-node creation first so `/dev/uio*` is already
  complete before the container starts
- expose only the required QAT-related device nodes to the container
- do not bind-mount the host `/dev` wholesale except as a short-lived debugging
  step

Example pattern for Docker-style launches:

```bash
DEVICE_ARGS=()
for dev in /dev/uio*; do
  DEVICE_ARGS+=(--device="${dev}:${dev}")
done

docker run --rm -it \
  "${DEVICE_ARGS[@]}" \
  -v /opt/wings-qat:/opt/wings-qat:ro \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/wings-qat/lib:/usr/local/lib \
  <your-image>
```

If the host only exposes a subset of UIO nodes inside the container, QATzip may
fail with errors such as `failed to open uio dev /dev/uio16`, `QZ_NO_HW`, or
`switch to SW`. In that case, check both:

- `/sys/class/uio/uio*/dev` on the host to confirm which UIO devices the kernel
  registered
- `/dev/uio*` inside the container to confirm the corresponding device nodes are
  actually present

Bind-mounting the host `/dev` into the container can be used as a temporary
validation shortcut, but it weakens isolation and exposes many unrelated
devices. The recommended long-term fix is still to repair host-side `udev` or
startup logic and then pass only the needed QAT device nodes through the
container runtime.

Default degradation policy:

- `build-wheel` treats QAT as optional on the common path.
- On `aarch64/arm64`, `kv-agent` is skipped by default.
- When the QAT toolchain or runtime libraries are missing, `kv-agent` is
  skipped by default and the LMCache wheel still builds.
- Set `WINGS_STRICT_QAT_BUILD=1` if you want optional-mode QAT build skips to
  become hard failures.

After `build-wheel` on `platform=common`, `dist/` always contains:

- `lmcache-*.whl`

When QAT prerequisites are available and the host is not ARM, `dist/` also
contains:

- `kv_agent-*.whl`

After `build-wheel --platform ascend`, `dist/` should contain:

- `lmcache-*.whl`

## Smoke Testing

For a minimal cold-start validation after installation, start from:

- `examples/lmcache-cold-start-smoke.yaml`

Suggested flow:

```bash
cd /harbor_data/wgd/wings-accel_lmcache_v2/LMCache_patch

# 1. Use the example config as your LMCache config input.
# 2. Send a few requests so local-disk payloads are materialized.
ls -l /tmp/lmcache-data
cat /tmp/lmcache-data/.wings_cold_start_manifest.json

# 3. Restart the process with the same local_disk path.
# 4. Check logs for a restore summary.
grep -E "Cold-start manifest restore finished|No Wings cold-start manifest found" <your-lmcache-log-file>
```

Expected cold-start success signal:

- log line containing `Cold-start manifest restore finished: restored=<n>` with `n > 0`

For QAT runtime readiness, run the bundled precheck first:

```bash
cd /harbor_data/wgd/wings-accel_lmcache_v2/LMCache_patch
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/wings-qat/lib:/usr/local/lib:${LD_LIBRARY_PATH}
/usr/bin/python3 scripts/precheck_qat_runtime.py
```

If you need the precheck to fail when no QAT devices are visible, use:

```bash
/usr/bin/python3 scripts/precheck_qat_runtime.py --require-devices
```

Expected QAT success signals:

- `kv_agent` imports successfully
- the precheck reports `qat_devices` with `available > 0`
- runtime logs contain `Wings QAT hook enabled`

If the precheck reports warnings but not failures, the common-path build can
still run with raw local-disk fallback.

For a minimal hardware-backed QAT smoke validation, start from:

- `examples/lmcache-qat-smoke.yaml`

Suggested flow:

```bash
cd /harbor_data/wgd/wings-accel_lmcache_v2/LMCache_patch

# 1. Bring the QAT devices up first. `state: down` means runtime validation
# will fall back instead of using QAT.
adf_ctl qat_dev0 up
adf_ctl qat_dev1 up
adf_ctl status

# 2. Ensure the kv_agent runtime can see both torch and QAT shared libraries.
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/wings-qat/lib:/usr/local/lib:${LD_LIBRARY_PATH}

# 3. Fail fast if QAT runtime prerequisites are still not satisfied.
/usr/bin/python3 scripts/precheck_qat_runtime.py --require-devices
```

Use the QAT config for your LMCache-enabled vLLM launch once the precheck
passes.

## Wings Config Shape

Wings-specific options live under `extra_config.wings` in the LMCache config.
Example:

```yaml
chunk_size: 256
local_cpu: true
local_disk: /data/lmcache
max_local_disk_size: 200
extra_config:
  wings:
    cold_start:
      enabled: true
      manifest_path: /data/lmcache/.wings_cold_start_manifest.json
      manifest_write_interval: 100
    maintenance:
      enabled: true
    full_sync:
      enabled: true
      default_reason: wings_manual_full_sync
    qat:
      enabled: true
      module_name: kv_agent
```

For `platform=ascend`, the first implementation keeps the old rule that QAT
must not be enabled.
