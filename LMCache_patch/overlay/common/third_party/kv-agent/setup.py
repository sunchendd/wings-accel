import os
import shutil
import fnmatch
import logging
from torch.utils import cpp_extension
from setuptools import setup, find_packages
from setuptools.command.install import install


def get_lib_files():
    lib_dir = "lib"
    lib_files = []
    patterns = ['*.so', '*.so.*', '*.a', '*.dll', '*.dylib']
    for root, _, files in os.walk(lib_dir):
        for file in files:
            for pat in patterns:
                if fnmatch.fnmatch(file, pat):
                    lib_files.append(os.path.join(root, file))
                    break
    logging.info("lib_files: %s", lib_files)
    return lib_files


class CustomInstall(install):
    def run(self):
        super().run()
        # 手动复制 lib 文件到安装目录
        lib_dir = os.path.join(self.install_lib, 'kv_agent', 'lib')
        os.makedirs(lib_dir, exist_ok=True)
        for lib_file in get_lib_files():
            shutil.copy2(lib_file, lib_dir)

setup(
    name="kv_agent",
    version="0.1",
    packages=find_packages(include=['kv_agent', 'kv_agent.*']),
    ext_modules=[
        cpp_extension.CppExtension(
            name="kv_agent._C",
            sources=["kv_agent.cpp"],
            library_dirs=["lib"],
            libraries=["qzip", "qatzip", "qat_s"],
            extra_compile_args=["-O3", "-fopenmp", "-mavx2", "-mavx512f", "-mavx512bw"],
            extra_link_args=["-lnuma"],
            runtime_library_dirs=["$ORIGIN/lib"],
        ),
    ],
    cmdclass={
        "build_ext": cpp_extension.BuildExtension,
        "install": CustomInstall,
    },
    package_data={"kv_agent": ["lib/*"]},  # 保证lib下所有文件都被打包
    include_package_data=True,
    install_requires=['torch'],
)