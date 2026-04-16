# Retained for backward compatibility with tools that do not fully support
# PEP 517/621 metadata resolution under --no-isolation builds.
from setuptools import find_packages, setup


setup(
    name="wings_engine_patch",
    version="1.0.0",
    description="Runtime monkey-patch framework for vLLM inference engines",
    packages=find_packages(include=["wings_engine_patch", "wings_engine_patch.*"]),
    package_data={"wings_engine_patch": ["supported_features.json"]},
    include_package_data=True,
    install_requires=["wrapt", "packaging"],
    extras_require={
        "vllm": [],
        "dev": ["pytest>=7", "pytest-cov", "build"],
    },
)
