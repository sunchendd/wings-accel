from setuptools import setup, find_packages

setup(
    name="wings_engine_patch",
    version="1.0.0",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "wrapt",
    ],
)
