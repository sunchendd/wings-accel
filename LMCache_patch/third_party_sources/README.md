# third_party_sources

This directory stores locked offline snapshots for non-LMCache source inputs
consumed by `LMCache_patch`.

Current planned use:

- `qatzip/`
  Common-path offline QATzip source snapshot consumed only when the build
  needs to materialize QATzip locally for `kv-agent`
- `kvcache-ops/`
  Ascend-only source snapshot consumed when building `--platform ascend`

These sources are treated the same way as the main LMCache tarball input:

- the archive content is locked in `manifest/lmcache.lock.json`
- the build workflow consumes only the locked snapshot
- live Git working trees are not used as build inputs
