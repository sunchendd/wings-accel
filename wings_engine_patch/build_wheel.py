import argparse
import base64
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from typing import Optional


DEFAULT_VERSION: str = "1.0.0"
PACKAGE_NAME = "wings_engine_patch"
PTH_FILENAME = "wings_engine_patch.pth"
PTH_BYTES = b"import wings_engine_patch._auto_patch\n"


def _clean_previous_builds(outdir: str) -> None:
    """Clean previous build artifacts."""
    paths_to_remove = ["build", "wings_engine_patch.egg-info"]
    if outdir == "dist":
        paths_to_remove.insert(0, "dist")
    for path in paths_to_remove:
        if os.path.exists(path):
            shutil.rmtree(path)


def _load_tomllib():
    try:
        import tomllib  # Python 3.11+

        return tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore

            return tomllib
        except ImportError:
            return None


def _read_version_with_tomllib(pyproject_path: str, tomllib_module) -> str:
    with open(pyproject_path, "rb") as pyproject_file:
        pyproject = tomllib_module.load(pyproject_file)
    return pyproject.get("project", {}).get("version", DEFAULT_VERSION)


def _extract_version_from_line(line: str) -> Optional[str]:
    stripped_line = line.strip()
    if not (stripped_line.startswith("version") and "=" in stripped_line):
        return None
    candidate = stripped_line.split("=", 1)[1].strip().strip('"').strip("'")
    return candidate if candidate else None


def _read_version_with_plaintext(pyproject_path: str) -> str:
    with open(pyproject_path, encoding="utf-8") as pyproject_file:
        for line in pyproject_file:
            version = _extract_version_from_line(line)
            if version is not None:
                return version
    return DEFAULT_VERSION


def _get_version_from_pyproject() -> str:
    """Extract version from pyproject.toml file."""
    pyproject_path = "pyproject.toml"
    if not os.path.exists(pyproject_path):
        return DEFAULT_VERSION

    tomllib_module = _load_tomllib()
    if tomllib_module is not None:
        return _read_version_with_tomllib(pyproject_path, tomllib_module)
    return _read_version_with_plaintext(pyproject_path)


def _remove_existing_wheels(outdir: str) -> None:
    for old_wheel in glob.glob(os.path.join(outdir, "*.whl")):
        os.remove(old_wheel)


def _build_base_wheel(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    _remove_existing_wheels(outdir)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            outdir,
        ]
    )
    whl_files = glob.glob(os.path.join(outdir, "*.whl"))
    if not whl_files:
        raise RuntimeError(f"No wheel file found in {outdir}/")
    return whl_files[0]


def _build_destination_path(version: str) -> str:
    data_dir = f"{PACKAGE_NAME}-{version}.data"
    return f"{data_dir}/purelib/{PTH_FILENAME}"


def _hash_record_entry(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _build_record_contents(
    *,
    record_name: str,
    old_record: str,
    destination_path: str,
    pth_bytes: bytes,
) -> str:
    stripped_record = "\n".join(
        line
        for line in old_record.splitlines()
        if line.strip() and not line.startswith(record_name)
    )
    return (
        stripped_record
        + f"\n{destination_path},{_hash_record_entry(pth_bytes)},{len(pth_bytes)}\n"
        + f"{record_name},,\n"
    )


def _copy_wheel_items_excluding_record(
    source_wheel: zipfile.ZipFile,
    dest_wheel: zipfile.ZipFile,
    record_name: str,
) -> None:
    dest_wheel.comment = source_wheel.comment
    for item in source_wheel.infolist():
        if item.filename == record_name:
            continue
        dest_wheel.writestr(item, source_wheel.read(item.filename))


def _repack_wheel_with_pth(whl_path: str, destination_path: str) -> None:
    new_whl_path = whl_path.replace(".whl", "_repacked.whl")
    with zipfile.ZipFile(whl_path, "r") as source_wheel:
        record_name = next(
            name for name in source_wheel.namelist() if name.endswith("/RECORD")
        )
        old_record = source_wheel.read(record_name).decode("utf-8")
        with zipfile.ZipFile(
            new_whl_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as repacked_wheel:
            _copy_wheel_items_excluding_record(source_wheel, repacked_wheel, record_name)
            repacked_wheel.writestr(destination_path, PTH_BYTES)
            repacked_wheel.writestr(
                record_name,
                _build_record_contents(
                    record_name=record_name,
                    old_record=old_record,
                    destination_path=destination_path,
                    pth_bytes=PTH_BYTES,
                ),
            )

    os.remove(whl_path)
    os.rename(new_whl_path, whl_path)


def build_wheel(outdir: str = "dist") -> None:
    _clean_previous_builds(outdir)
    whl_path = _build_base_wheel(outdir)
    print(f"Original wheel: {whl_path}")

    destination_path = _build_destination_path(_get_version_from_pyproject())
    print(f"Adding {PTH_FILENAME} to {destination_path}")
    _repack_wheel_with_pth(whl_path, destination_path)
    print(f"Modified wheel: {whl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build wings_engine_patch wheel with .pth injection."
    )
    parser.add_argument(
        "--outdir",
        default="dist",
        help="Directory where the built wheel will be written.",
    )
    args = parser.parse_args()
    build_wheel(outdir=args.outdir)
