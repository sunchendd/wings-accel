# LMCache Patch-First Guidelines

## Scope

These instructions apply to work under `wings-accel/LMCache_patch/`.

## Architecture

- This tree is a patch-first integration workspace, not a normal editable LMCache checkout.
- Treat these paths as source of truth:
  - `manifest/`
  - `overlay/common/`
  - `overlay/ascend/`
  - `patches/common/<version>/`
  - `patches/ascend/<version>/`
  - `scripts/`
  - `install.py`
- Treat these paths as generated or disposable output:
  - `build/generated/LMCache/`
  - `dist/`
  - `__pycache__/`
- Keep `upstream_sources/` pristine.
- Keep `third_party_sources/` as locked offline source snapshots, not live Git worktrees.

## Workflow

- Use `python3 install.py <command>` as the entrypoint for this subtree.
- Common workflow:
  - `python3 install.py verify-upgrade`
  - `python3 install.py build-wheel`
- Ascend workflow:
  - `python3 install.py prepare-ascend-sources`
  - `python3 install.py verify-upgrade --platform ascend`
  - `python3 install.py build-wheel --platform ascend`
- Use `python3 install.py build-kv-agent` only when debugging the QAT extension directly.
- Use `python3 install.py regen-patchset` after intentional upstream-side edits needed to refresh patch files.
- Never treat `build/generated/LMCache/` as the long-term source of truth. Put Wings-owned logic in `overlay/` and keep upstream edits as small patches in `patches/`.

## Testing

- Run Wings regression tests from this directory with:
  - `pytest -xvs overlay/common/tests_wings`
  - `pytest -xvs overlay/ascend/tests_wings`
- Keep test coverage aligned to the requirement families tracked in `STATUS.md`.
- When changing hook behavior, add or update the matching Wings tests in the same requirement area.

## Conventions

- Keep patch files version-scoped and minimal.
- Prefer implementing new behavior in `overlay/common` or `overlay/ascend`; use patch hunks only to hook upstream entrypoints.
- Update `manifest/lmcache.lock.json` when changing the upstream LMCache tarball.
- Only `prepare-ascend-sources` may rely on the internal Git path; replay and build steps must work from locked local sources.
- Ascend reuses the common cold-start and local-disk lifecycle path, but does not support QAT.
- Update `STATUS.md` when requirement coverage or migration status changes.