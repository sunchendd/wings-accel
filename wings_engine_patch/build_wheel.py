import os
import shutil
import glob
import subprocess
import zipfile

def build_wheel():
    # Clean previous builds
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("wings_engine_patch.egg-info"):
        shutil.rmtree("wings_engine_patch.egg-info")

    # Build the wheel
    subprocess.check_call(["python3", "setup.py", "bdist_wheel"])

    # Find the built wheel
    whl_files = glob.glob("dist/*.whl")
    if not whl_files:
        raise Exception("No wheel file found in dist/")
    whl_path = whl_files[0]
    print(f"Original wheel: {whl_path}")

    # Define the name of the folder inside the wheel's .data directory
    # For a pure python package, it's usually 'purelib'.
    # We need to construct the path 'wings_engine_patch-{version}.data/purelib/wings_engine_patch.pth'
    
    # Get version from setup.py (hardcoded here based on previous file content)
    version = "1.0.0" 
    package_name = "wings_engine_patch"
    data_dir = f"{package_name}-{version}.data"
    destination_path = f"{data_dir}/purelib/wings_engine_patch.pth"
    
    # We need to repack the wheel. Python's zipfile module supports append 'a', but modifying paths is tricky.
    # It's safer to read all files, and write a new zip.
    
    new_whl_path = whl_path.replace(".whl", "_repacked.whl")
    
    with zipfile.ZipFile(whl_path, 'r') as zin:
        with zipfile.ZipFile(new_whl_path, 'w') as zout:
            zout.comment = zin.comment # preserve the comment
            for item in zin.infolist():
                zout.writestr(item, zin.read(item.filename))
            
            # Add wings_engine_patch.pth
            print(f"Adding wings_engine_patch.pth to {destination_path}")
            # Dynamically write the content of .pth file
            pth_content = "import wings_engine_patch._auto_patch\n"
            zout.writestr(destination_path, pth_content)

    # Replace original wheel
    os.remove(whl_path)
    os.rename(new_whl_path, whl_path)
    print(f"Modified wheel: {whl_path}")

if __name__ == "__main__":
    build_wheel()
