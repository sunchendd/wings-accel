# Wings Engine Patch

Wings Engine Patch is a lightweight, dynamic patching framework designed to inject feature enhancements and fixes into inference engines (specifically `vllm` and `vllm_ascend`) at runtime, without modifying the engine's source code directly.

## Key Features

*   **Non-Intrusive**: Patches are applied at runtime using Python's import hooks and `wrapt`, ensuring the original package installation remains pristine.
*   **Version Controlled**: Patches are strictly scoped to specific engine versions (e.g., `vllm_ascend` version `0.12.0rc1`).
*   **Feature-Based Management**: Patches are grouped into named "features" (e.g., `soft_fp8`, `soft_fp4`). Users enable features, not individual files.
*   **Intelligent Dependency Resolution**:
    *   **Shared Patches**: If multiple features rely on the same underlying patch implementation, enabling one automatically activates the shared components.
    *   **Deduplication**: The engine ensures that any specific patch function is executed exactly once, regardless of how many enabled features require it.

## Installation

This project uses a custom build process to inject a `.pth` entry-point hook into the wheel.

**Recommended — use the top-level Makefile from the repository root:**

```bash
make dev-setup   # first time: create .venv + install build/test deps
make build       # build the wheel (output: dist/*.whl)
make install     # build + pip install into current environment
```

**Or use the install CLI directly:**

```bash
python install.py --features '{"vllm_ascend": {"version": "0.12.0rc1", "features": ["soft_fp8"]}}'
```

**Manual (advanced):**

```bash
cd wings_engine_patch
python build_wheel.py          # generates dist/wings_engine_patch-*.whl
pip install dist/wings_engine_patch-*.whl --force-reinstall
```

## Usage

Enable patches by setting the `WINGS_ENGINE_PATCH_OPTIONS` environment variable before running your application. The variable accepts a JSON string defining the target engine, its version, and the list of features to enable.

### Configuration Format

```json
{
    "engine_name": {
        "version": "x.y.z",
        "features": ["feature_name_1", "feature_name_2"]
    }
}
```

### Example

To enable `soft_fp8` support for `vllm_ascend` version `0.12.0rc1`:

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{
    "vllm_ascend": {
        "version": "0.12.0rc1",
        "features": ["soft_fp8"]
    }
}'

python3 -m vllm.entrypoints.api_server ...
```

> **Note**: If the configured version matches the installed engine version, patches are applied. If there is a mismatch, the system may attempt to fall back to a default version configuration if one is defined in the registry.

## Development

### Project Structure

```text
wings_engine_patch/
├── wings_engine_patch/
│   ├── _auto_patch.py       # Entry point triggered by .pth file
│   ├── registry.py          # Central registry mapping features to patch functions
│   └── patch_<engine>/      # Actual patch implementations
│       └── <version>/
├── build_wheel.py           # Custom build script
└── setup.py
```

### Adding a New Patch

1.  **Implement the Patch**: Create a Python module (e.g., `patch_my_feature.py`) inside the appropriate engine/version directory.
    *   Define a patch function (e.g., `patch_SpecificFunction`).
    *   Use `wrapt.register_post_import_hook` to safely patch modules *after* they are imported.

2.  **Register the Patch**: Update `wings_engine_patch/registry.py`.
    *   Import your patch function inside the appropriate version builder function (e.g., `_build_vllm_ascend_v0_12_0rc1_features`).
    *   Add the patch function object directly to the feature list.

### Critical Patch Development Rules & Best Practices

1.  **Lazy Imports (No Module-Level Engine Imports)**:
    *   **Rule**: In your patch implementation files, **never** import engine modules (e.g., `vllm`, `vllm_ascend`, `torch`) at the top module level.
    *   **Implementation**: All imports of engine-related modules **must** be placed inside the patch function or the wrapper function where they are actually used.
    *   **Reason**: Early imports can trigger the loading of the target module before the patching mechanism is ready (preventing the patch from registering), or cause circular dependency errors.

2.  **Merged Patches for Shared Targets**:
    *   **Rule**: If multiple features need to modify the *same* target function, you must consolidate their logic into a **single** patch function.
    *   **Implementation**: Assign this single shared patch function to both features in the registry. The system's "propagation" mechanism will ensure that enabling either feature automatically enables the other, keeping them in sync.
    *   **Reason**: This completely avoids undefined behavior related to the order in which multiple patches might be applied to the same function.

3.  **Strict Use of Post-Import Hooks**:
    *   **Rule**: Patches must be applied using `wrapt.register_post_import_hook` to safely patch modules *after* they are imported.
    *   **Reason**: This ensures the original package installation remains pristine and allows patching at runtime dynamically.

4.  **Idempotency**:
    *   **Rule**: Patch logic must be idempotent.
    *   **Implementation**: Ensure that applying the patch multiple times does not result in errors or duplicated logic (e.g., wrapping the same function multiple times).
    *   **Reason**: Adds a layer of safety on top of the registry's deduplication mechanism.

5.  **Granularity**:
    *   Create small, specific patch functions (e.g., `patch_AscendQuantConfig_get_quant_method`) rather than monolithic "patch all" functions.

### Feature Propagation and Patch Deduplication

When multiple features share the same patch function, enabling any one of those features will automatically enable all features that share that patch (unless the patch is explicitly marked as `non_propagating`).

- **Propagation**: By default, if two or more features reference the same patch function, enabling one feature will cause all features sharing that patch to be enabled. This ensures that shared logic is always applied consistently and avoids partial/inconsistent patching.
- **Non-Propagating Patches**: If a patch should not trigger feature propagation, add it to the `non_propagating_patches` set in the version builder. These patches will not cause other features to be auto-enabled.
- **Deduplication**: Each patch function is executed at most once, even if it is referenced by multiple features.

**Example:**

If both `soft_fp8` and `soft_fp4` reference `patch_utils.patch_ASCEND_QUANTIZATION_METHOD_MAP`, enabling either feature will result in both being enabled, unless that patch is listed in `non_propagating_patches`.
