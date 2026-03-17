import os
import sys
import hashlib
import base64
import shutil
import glob
import subprocess
import zipfile
import argparse


def _clean_previous_builds(outdir: str) -> None:
    """Clean previous build artifacts."""
    if outdir == "dist" and os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("wings_engine_patch.egg-info"):
        shutil.rmtree("wings_engine_patch.egg-info")


def _get_version_from_pyproject() -> str:
    """Extract version from pyproject.toml file."""
    version = "1.0.0"  # fallback
    pyproject_path = "pyproject.toml"
    
    if not os.path.exists(pyproject_path):
        return version
    
    # Try to import tomllib or tomli
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            tomllib = None
    
    if tomllib:
        with open(pyproject_path, "rb") as tf:
            pyproject = tomllib.load(tf)
            version = pyproject.get("project", {}).get("version", version)
    else:
        # Plain string search fallback
        with open(pyproject_path) as tf:
            for line in tf:
                line = line.strip()
                if line.startswith("version") and "=" in line:
                    candidate = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if candidate:
                        version = candidate
                        break
    return version


def build_wheel(outdir="dist"):
    # Clean previous builds
    _clean_previous_builds(outdir)
    
    os.makedirs(outdir, exist_ok=True)
    for old_wheel in glob.glob(os.path.join(outdir, "*.whl")):
        os.remove(old_wheel)

    # Build the wheel using current interpreter (supports venv)
    subprocess.check_call([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", outdir])

    # Find the built wheel
    whl_files = glob.glob(os.path.join(outdir, "*.whl"))
    if not whl_files:
        raise Exception(f"No wheel file found in {outdir}/")
    whl_path = whl_files[0]
    print(f"Original wheel: {whl_path}")

    # Define the name of the folder inside the wheel's .data directory
    # For a pure python package, it's usually 'purelib'.
    # We need to construct the path 'wings_engine_patch-{version}.data/purelib/wings_engine_patch.pth'
    
    # Get version dynamically from pyproject.toml
    version = _get_version_from_pyproject()
    
    package_name = "wings_engine_patch"
    data_dir = f"{package_name}-{version}.data"
    destination_path = f"{data_dir}/purelib/wings_engine_patch.pth"
    
    # We need to repack the wheel. Python's zipfile module supports append 'a', but modifying paths is tricky.
    # It's safer to read all files, and write a new zip.
    
    new_whl_path = whl_path.replace(".whl", "_repacked.whl")
    pth_content = "import wings_engine_patch._auto_patch\n"
    pth_bytes = pth_content.encode("utf-8")
    pth_hash = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(pth_bytes).digest()).rstrip(b"=").decode("ascii")

    with zipfile.ZipFile(whl_path, 'r') as zin:
        record_name = next(n for n in zin.namelist() if n.endswith("/RECORD"))
        old_record = zin.read(record_name).decode("utf-8")
        with zipfile.ZipFile(new_whl_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            zout.comment = zin.comment
            for item in zin.infolist():
                if item.filename == record_name:
                    continue  # will rewrite RECORD below
                zout.writestr(item, zin.read(item.filename))

            # Add wings_engine_patch.pth
            print(f"Adding wings_engine_patch.pth to {destination_path}")
            zout.writestr(destination_path, pth_bytes)

            # Update RECORD with new entry + pth entry.
            # The original RECORD already ends with "<record_name>,,"  (pip
            # always writes that sentinel last).  Strip it before appending so
            # we don't produce a duplicate RECORD entry.
            stripped_record = "\n".join(
                line for line in old_record.splitlines()
                if line.strip() and not line.startswith(record_name)
            )
            new_record = (
                stripped_record
                + f"\n{destination_path},{pth_hash},{len(pth_bytes)}\n"
                + f"{record_name},,\n"
            )
            zout.writestr(record_name, new_record)

    # Replace original wheel
    os.remove(whl_path)
    os.rename(new_whl_path, whl_path)
    print(f"Modified wheel: {whl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build wings_engine_patch wheel with .pth injection.")
    parser.add_argument("--outdir", default="dist", help="Directory where the built wheel will be written.")
    args = parser.parse_args()
    build_wheel(outdir=args.outdir)
