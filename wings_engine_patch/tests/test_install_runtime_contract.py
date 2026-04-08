import io
import sys
from contextlib import suppress

import install as install_module


def test_find_local_whl_prefers_flat_delivery_dir(tmp_path, monkeypatch):
    wheel_in_flat_delivery = tmp_path / "wings_engine_patch-1.2.3-py3-none-any.whl"
    wheel_in_flat_delivery.write_text("wheel", encoding="utf-8")

    monkeypatch.setattr(install_module, "_BASE_DIR", tmp_path)
    monkeypatch.setattr(install_module, "_LOCAL_WHEEL_DIR", tmp_path / "build" / "output")

    assert install_module._find_local_whl() == wheel_in_flat_delivery  # pylint: disable=protected-access


def test_find_local_whl_reads_build_output_when_delivery_is_flat_source_tree(tmp_path, monkeypatch):
    wheel_in_build_output = tmp_path / "build" / "output" / "wings_engine_patch-1.2.3-py3-none-any.whl"
    wheel_in_build_output.parent.mkdir(parents=True)
    wheel_in_build_output.write_text("wheel", encoding="utf-8")

    monkeypatch.setattr(install_module, "_BASE_DIR", tmp_path)
    monkeypatch.setattr(install_module, "_LOCAL_WHEEL_DIR", wheel_in_build_output.parent)

    assert install_module._find_local_whl() == wheel_in_build_output  # pylint: disable=protected-access


def test_find_local_runtime_dep_wheel_in_flat_delivery(tmp_path, monkeypatch):
    wrapt_wheel = tmp_path / "wrapt-2.1.2-py3-none-any.whl"
    wrapt_wheel.write_text("wheel", encoding="utf-8")

    monkeypatch.setattr(install_module, "_BASE_DIR", tmp_path)
    monkeypatch.setattr(install_module, "_LOCAL_WHEEL_DIR", tmp_path / "build" / "output")

    assert install_module._find_local_wheel_by_prefix("wrapt") == wrapt_wheel  # pylint: disable=protected-access


def test_find_local_runtime_dep_wheel_in_build_output(tmp_path, monkeypatch):
    packaging_wheel = tmp_path / "build" / "output" / "packaging-26.0-py3-none-any.whl"
    packaging_wheel.parent.mkdir(parents=True)
    packaging_wheel.write_text("wheel", encoding="utf-8")

    monkeypatch.setattr(install_module, "_BASE_DIR", tmp_path)
    monkeypatch.setattr(install_module, "_LOCAL_WHEEL_DIR", packaging_wheel.parent)

    assert install_module._find_local_wheel_by_prefix("packaging") == packaging_wheel  # pylint: disable=protected-access


def test_offline_local_wheel_install_does_not_force_reinstall(monkeypatch, tmp_path):
    wheel_path = tmp_path / "wings_engine_patch-1.0.0-py3-none-any.whl"
    wheel_path.write_text("wheel", encoding="utf-8")
    commands = []

    monkeypatch.setattr(
        install_module,
        "_find_local_wheel_by_prefix",
        lambda prefix: wheel_path if prefix == "wings_engine_patch" else None,
    )
    monkeypatch.setattr(install_module, "_find_local_whl", lambda: wheel_path)
    monkeypatch.setattr(install_module, "_has_local_runtime_deps", lambda: False)
    monkeypatch.setattr(install_module, "_install_local_feature_wheels", lambda *args, **kwargs: None)
    monkeypatch.setattr(install_module, "_print_env_hint", lambda *args, **kwargs: None)
    monkeypatch.setattr(install_module.subprocess, "check_call", lambda cmd: commands.append(cmd))

    install_module.install_engine("vllm", "0.17.0", ["ears"])

    cmd = commands[0]
    assert "--force-reinstall" not in " ".join(cmd)


def test_future_vllm_version_warns_and_preserves_requested_public_features(monkeypatch):
    calls = []
    captured_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured_stderr)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install.py",
            "--dry-run",
            "--features",
            '{"vllm": {"version": "0.18.0", "features": ["ears"]}}',
        ],
    )
    monkeypatch.setattr(install_module, "install_runtime_dependencies", lambda dry_run=False: None)
    monkeypatch.setattr(
        install_module,
        "install_engine",
        lambda engine_name, version, features, dry_run=False: calls.append(
            (engine_name, version, features, dry_run)
        ),
    )

    with suppress(SystemExit):
        install_module.main()

    assert calls == [("vllm", "0.17.0", ["ears"], True)]
    assert "newer than the highest validated version" in captured_stderr.getvalue()


def test_install_runtime_dependencies_installs_wrapt_packaging_then_best_effort_arctic(
    monkeypatch, tmp_path
):
    wrapt_wheel = tmp_path / "wrapt-2.1.2-py3-none-any.whl"
    packaging_wheel = tmp_path / "packaging-26.0-py3-none-any.whl"
    calls = []

    monkeypatch.setattr(
        install_module,
        "_find_local_wheel_by_prefix",
        lambda prefix: {
            "wrapt": wrapt_wheel,
            "packaging": packaging_wheel,
        }.get(prefix),
    )
    monkeypatch.setattr(
        install_module,
        "_install_local_dependency",
        lambda package_name, module_name, wheel_path, dry_run=False, **kwargs: calls.append(
            (
                "dep",
                package_name,
                module_name,
                wheel_path,
                dry_run,
                kwargs.get("missing_ok", False),
            )
        ),
    )
    monkeypatch.setattr(
        install_module,
        "_install_arctic_inference",
        lambda dry_run=False: calls.append(("arctic", dry_run)),
    )

    install_module.install_runtime_dependencies(dry_run=True)

    assert calls == [
        ("dep", "wrapt", "wrapt", wrapt_wheel, True, False),
        ("dep", "packaging", "packaging", packaging_wheel, True, False),
        ("arctic", True),
    ]


def test_arctic_inference_runtime_dependency_remains_best_effort(monkeypatch):
    calls = []

    monkeypatch.setattr(install_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(install_module, "_find_arctic_inference_whl", lambda: None)

    def fake_install_local_dependency(package_name, module_name, wheel_path, dry_run=False, **kwargs):
        calls.append(
            {
                "package_name": package_name,
                "module_name": module_name,
                "wheel_path": wheel_path,
                "dry_run": dry_run,
                "no_deps": kwargs.get("no_deps"),
                "missing_ok": kwargs.get("missing_ok"),
            }
        )

    monkeypatch.setattr(install_module, "_install_local_dependency", fake_install_local_dependency)

    install_module._install_arctic_inference()  # pylint: disable=protected-access

    assert calls == [
        {
            "package_name": "arctic-inference",
            "module_name": "arctic_inference",
            "wheel_path": None,
            "dry_run": False,
            "no_deps": True,
            "missing_ok": True,
        }
    ]


def test_find_arctic_inference_wheel_supports_flat_delivery_variants(tmp_path, monkeypatch):
    arctic_wheel = tmp_path / "arctic-inference-1.0.0-py3-none-any.whl"
    arctic_wheel.write_text("wheel", encoding="utf-8")

    monkeypatch.setattr(install_module, "_BASE_DIR", tmp_path)
    monkeypatch.setattr(install_module, "_LOCAL_WHEEL_DIR", tmp_path / "build" / "output")

    assert install_module._find_arctic_inference_whl() == arctic_wheel  # pylint: disable=protected-access
