"""
Integration test for the wings-engine-patch monkey-patch framework.

Verifies:
  1. .pth hook is installed in site-packages
  2. wrapt post-import hook mechanism works (synthetic module)
  3. the hello_world diagnostic patch can be enabled cleanly for vllm
  4. _auto_patch.py is triggered by the .pth file on Python startup (subprocess)

Run with:
    python tests/test_integration_real.py
or:
    pytest tests/test_integration_real.py -v
"""

import importlib
import importlib.util
import os
import site
import subprocess
import sys
import textwrap
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_python(code: str, env_extra: dict = None) -> tuple[int, str, str]:
    """Run a Python snippet in a clean subprocess, return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# 1. .pth hook presence
# ---------------------------------------------------------------------------

class TestPthHookInstalled(unittest.TestCase):
    def test_pth_file_exists_in_site_packages(self):
        sp = site.getsitepackages()[0]
        pth_path = os.path.join(sp, "wings_engine_patch.pth")
        self.assertTrue(
            os.path.exists(pth_path),
            f"wings_engine_patch.pth not found in {sp}. Did you install the whl?",
        )

    def test_pth_content_imports_auto_patch(self):
        sp = site.getsitepackages()[0]
        pth_path = os.path.join(sp, "wings_engine_patch.pth")
        if not os.path.exists(pth_path):
            self.skipTest("pth file not installed")
        content = open(pth_path).read().strip()
        self.assertEqual(content, "import wings_engine_patch._auto_patch",
                         f"Unexpected .pth content: {content!r}")


# ---------------------------------------------------------------------------
# 2. wrapt hook mechanism (synthetic)
# ---------------------------------------------------------------------------

class TestWraptHookMechanism(unittest.TestCase):
    """
    Verify that wrapt.register_post_import_hook fires correctly.

    Note: wrapt hooks fire through Python's import machinery.  Manually
    inserting a module object into sys.modules bypasses that machinery, so
    hooks do NOT fire that way.  The correct test approach is to use
    wrapt.notify_module_loaded() — the official API for triggering hooks
    on an already-present module — or to let the real import system load
    the module.
    """

    def setUp(self):
        import wrapt
        self.wrapt = wrapt
        sys.modules.pop("_wings_test_synthetic_mod", None)
        sys.modules.pop("_wings_test_synthetic_mod2", None)

    def test_hook_fires_via_notify_module_loaded(self):
        """
        notify_module_loaded() is the correct way to trigger all registered
        hooks for a module that was inserted into sys.modules directly.
        This is how wrapt itself handles already-loaded modules.
        """
        fired = []

        def my_hook(module):
            fired.append(module.__name__)

        self.wrapt.register_post_import_hook(my_hook, "_wings_test_synthetic_mod")

        mod = types.ModuleType("_wings_test_synthetic_mod")
        mod.value = 42
        sys.modules["_wings_test_synthetic_mod"] = mod
        # Trigger all pending hooks for this module
        self.wrapt.notify_module_loaded(mod)

        self.assertIn("_wings_test_synthetic_mod", fired,
                      "wrapt hook did not fire via notify_module_loaded")

    def test_hook_can_mutate_module_via_notify(self):
        """Hook can mutate module attributes when triggered via notify_module_loaded."""
        import wrapt

        def patch_it(module):
            module.patched_attr = "PATCHED"

        wrapt.register_post_import_hook(patch_it, "_wings_test_synthetic_mod2")
        mod = types.ModuleType("_wings_test_synthetic_mod2")
        mod.original_attr = "ORIGINAL"
        sys.modules["_wings_test_synthetic_mod2"] = mod
        wrapt.notify_module_loaded(mod)

        self.assertEqual(mod.patched_attr, "PATCHED",
                         "Hook did not mutate module via notify_module_loaded")

    def test_hook_fires_on_real_import(self):
        """
        Hooks also fire during a real import (not sys.modules injection).
        Verified using a module that is guaranteed to not be cached yet.
        We use importlib to reload a lightweight stdlib module to confirm.
        """
        import wrapt
        fired = []

        # Use 'colorsys' — a tiny stdlib module unlikely to be already loaded
        sys.modules.pop("colorsys", None)

        wrapt.register_post_import_hook(lambda m: fired.append(m.__name__), "colorsys")
        import colorsys  # noqa — triggers real import  # noqa: F401

        self.assertIn("colorsys", fired,
                      "wrapt hook did not fire during real import of colorsys")

    def tearDown(self):
        sys.modules.pop("_wings_test_synthetic_mod", None)
        sys.modules.pop("_wings_test_synthetic_mod2", None)


# ---------------------------------------------------------------------------
# 3. Direct patch registration: hello_world
# ---------------------------------------------------------------------------

class TestHelloWorldPatch(unittest.TestCase):

    def test_patch_vllm_hello_world_logs_to_stderr(self):
        import io
        from unittest.mock import patch

        from wings_engine_patch.patch_vllm_container.v0_12_0_empty import hello_world_patch

        buf = io.StringIO()
        with patch("sys.stderr", buf):
            hello_world_patch.patch_vllm_hello_world()

        self.assertIn("[Vllm Patch] hello_world patch enabled", buf.getvalue())


# ---------------------------------------------------------------------------
# 4. Subprocess: _auto_patch.py triggered by .pth on startup
# ---------------------------------------------------------------------------

class TestAutoPatchSubprocess(unittest.TestCase):
    """
    Starts a fresh Python process with WINGS_ENGINE_PATCH_OPTIONS set.
    The .pth file should trigger _auto_patch.py automatically.
    Checks:
      - No Critical Error on stderr
      - Patch hooks are registered (wrapt hooks are set up)
    """

    HELLO_WORLD_OPTIONS = '{"vllm": {"version": "0.12.0+empty", "features": ["hello_world"]}}'
    HELLO_WORLD_LOG = '[Vllm Patch] hello_world patch enabled'
    HELLO_WORLD_WARNING = "Feature 'hello_world' not found in registry"
    PATCH_FAILURE_LOG = '[Wings Engine Patch] Patch failed'
    PATCH_EXECUTION_ERROR_LOG = '[Wings Engine Patch] Error executing patch'

    def test_auto_patch_no_critical_error(self):
        code = "print('auto_patch_loaded')"
        rc, stdout, stderr = _run_python(
            code,
            env_extra={"WINGS_ENGINE_PATCH_OPTIONS": self.HELLO_WORLD_OPTIONS},
        )
        self.assertNotIn(
            "[Wings Engine Patch] Critical Error",
            stderr,
            f"Critical error found in stderr:\n{stderr}",
        )
        self.assertIn("auto_patch_loaded", stdout,
                      f"auto_patch did not complete. stdout={stdout!r} stderr={stderr!r}")
        print(f"\n  [OK] _auto_patch ran without critical error")

    def test_auto_patch_missing_version_warns(self):
        """Missing 'version' key in config should produce a warning, not crash."""
        bad_options = '{"vllm": {"features": ["hello_world"]}}'
        code = "import wings_engine_patch._auto_patch; print('ok')"
        rc, stdout, stderr = _run_python(
            code,
            env_extra={"WINGS_ENGINE_PATCH_OPTIONS": bad_options},
        )
        self.assertIn("Warning", stderr,
                      f"Expected warning for missing version, got stderr:\n{stderr}")
        self.assertEqual(rc, 0, "Process should not crash on bad config")

    def test_auto_patch_bad_json_warns(self):
        """Malformed JSON should warn but not crash."""
        code = "import wings_engine_patch._auto_patch; print('ok')"
        rc, stdout, stderr = _run_python(
            code,
            env_extra={"WINGS_ENGINE_PATCH_OPTIONS": "{bad json}"},
        )
        self.assertIn("Warning", stderr)
        self.assertEqual(rc, 0)

    def test_auto_patch_unknown_engine_warns(self):
        """Unknown engine name should warn but not crash."""
        code = "import wings_engine_patch._auto_patch; print('ok')"
        rc, stdout, stderr = _run_python(
            code,
            env_extra={"WINGS_ENGINE_PATCH_OPTIONS": '{"unknown_engine": {"version": "1.0", "features": ["x"]}}'},
        )
        self.assertIn("Warning", stderr)
        self.assertEqual(rc, 0)

    def test_auto_patch_hello_world_logs_on_startup(self):
        """hello_world should log to stderr when auto-patch enables it at startup."""
        code = "print('startup_probe')"
        rc, stdout, stderr = _run_python(
            code,
            env_extra={"WINGS_ENGINE_PATCH_OPTIONS": self.HELLO_WORLD_OPTIONS},
        )
        self.assertEqual(rc, 0, f"Process should not crash. stdout={stdout!r} stderr={stderr!r}")
        self.assertIn("startup_probe", stdout)
        self.assertNotIn(
            self.HELLO_WORLD_WARNING,
            stderr,
            f"hello_world should be registered during startup, not rejected as missing:\n{stderr}",
        )
        self.assertNotIn(
            self.PATCH_FAILURE_LOG,
            stderr,
            f"hello_world startup should not report patch failures:\n{stderr}",
        )
        self.assertNotIn(
            self.PATCH_EXECUTION_ERROR_LOG,
            stderr,
            f"hello_world startup should not report patch execution errors:\n{stderr}",
        )
        self.assertIn(
            self.HELLO_WORLD_LOG,
            stderr,
            f"Expected hello_world startup log in stderr, got:\n{stderr}",
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
