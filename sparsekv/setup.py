import os
import subprocess
from dataclasses import dataclass
import sys
import shutil
import zipfile

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


@dataclass(frozen=True)
class CMakeProject:
    cmake_lists_dir: str
    build_subdir: str


def _is_ninja_available() -> bool:
    """检查Ninja构建工具是否可用"""
    try:
        subprocess.check_output(["ninja", "--version"])
        return True
    except Exception:
        return False


# 第一步：负责将c库编译.so
class CMakeBuildExt(build_ext):
    configured: set[str] = set()

    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e
        super().run()

    def build_extension(self, ext: Extension) -> None:
        cmake_project = getattr(ext, "cmake_project", None)
        if cmake_project is None:
            super().build_extension(ext)
            return

        # 初始化编译配置
        cfg, build_temp = self._init_build_config(cmake_project)
        # 构造cmake参数
        cmake_args = self._get_cmake_args(cfg)
        # 执行cmake配置和编译
        self._execute_cmake_build(cmake_project, cfg, build_temp, cmake_args)
        # 查找并拷贝so文件
        self._find_and_copy_so_files(build_temp)

    def _init_build_config(self, cmake_project):
        """初始化编译配置参数"""
        cfg = "Debug" if self.debug else "RelWithDebInfo"
        build_temp = os.path.abspath(os.path.join(self.build_temp, cmake_project.build_subdir))
        os.makedirs(build_temp, exist_ok=True)
        return cfg, build_temp

    def _get_cmake_args(self, cfg):
        """构造cmake编译参数"""
        runtime_environment = os.environ.get("RUNTIME_ENVIRONMENT", "cuda")
        logger_backend = os.environ.get("LOGGER_BACKEND", "spdlog")
        download_dependence = os.environ.get("DOWNLOAD_DEPENDENCE", "ON")

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DRUNTIME_ENVIRONMENT={}".format(runtime_environment),
            "-DLOGGER_BACKEND={}".format(logger_backend),
            "-DDOWNLOAD_DEPENDENCE={}".format(download_dependence),
            "-DPython_EXECUTABLE={}".format(sys.executable),
        ]
        return [a for a in cmake_args if a]

    def _execute_cmake_command(self, cmd, error_msg):
        """执行cmake命令并处理异常"""
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{error_msg}: {e.output}") from e

    def _execute_cmake_build(self, cmake_project, cfg, build_temp, cmake_args):
        """执行cmake配置和编译流程"""
        # CMake配置
        generator = ["-G", "Ninja"] if _is_ninja_available() else []
        cmake_cmd = ["cmake", "-S", cmake_project.cmake_lists_dir, "-B", build_temp, *generator, *cmake_args]
        self._execute_cmake_command(cmake_cmd, "cmake config failed")

        # 执行编译
        num_jobs = int(os.environ.get("MAX_JOBS") or (os.cpu_count() or 1))
        build_cmd = ["cmake", "--build", build_temp, "--config", cfg, "-j", str(num_jobs)]
        self._execute_cmake_command(build_cmd, "build failed")

    def _scan_files_for_so(self, search_path, target_prefixes):
        """扫描单个路径下的SO文件"""
        so_files = []
        for root, _, files in os.walk(search_path):
            for file in files:
                if self._is_target_so_file(file, target_prefixes):
                    so_path = os.path.join(root, file)
                    so_files.append((so_path,))
        return so_files

    def _is_target_so_file(self, file_name, target_prefixes):
        """判断文件是否为目标SO文件"""
        return file_name.endswith(".so") and any(file_name.startswith(p) for p in target_prefixes)

    def _find_so_files(self, build_temp):
        """查找目标so文件"""
        script_dir = os.path.abspath(os.path.dirname(__file__))
        search_paths = [build_temp, os.path.join(script_dir, "native")]
        so_files = []
        target_prefixes = ["_local_kvstore", "_offload_ops", "_paged_kmeans", "_prefetch_engine"]

        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
            # 拆分嵌套循环到子函数，降低代码深度
            path_so_files = self._scan_files_for_so(search_path, target_prefixes)
            so_files.extend(path_so_files)
        return so_files

    def _copy_so_file(self, so_path, dest_dir):
        """拷贝单个so文件（去重）"""
        so_filename = os.path.basename(so_path)
        dest_path = os.path.join(dest_dir, so_filename)
        if so_path != dest_path:
            shutil.copy2(so_path, dest_dir)

    def _find_and_copy_so_files(self, build_temp):
        """查找并拷贝so文件到指定目录"""
        so_files = self._find_so_files(build_temp)
        if not so_files:
            return

        script_dir = os.path.abspath(os.path.dirname(__file__))
        native_dir = os.path.join(script_dir, "native")
        build_lib_dir = os.path.join(script_dir, "build", "lib", "vsparse", "native")
        os.makedirs(build_lib_dir, exist_ok=True)

        for so_path, *_ in so_files:
            # 拷贝到native目录
            self._copy_so_file(so_path, native_dir)
            # 拷贝到build/lib目录
            self._copy_so_file(so_path, build_lib_dir)


# 在whl生成后自动合并.so，到新的whl包
class BdistWheelWithSO(_bdist_wheel):
    def run(self):
        # 先执行原生bdist_wheel，生成基础whl（仅含.py）
        super().run()

        # 合并.so到生成的whl中
        project_root = os.path.abspath(os.path.dirname(__file__))
        dist_dir = os.path.join(project_root, self.dist_dir)

        # 找到刚生成的whl文件
        whl_files = [f for f in os.listdir(dist_dir) if f.endswith(".whl") and f.startswith("vsparse-")]
        if not whl_files:
            raise RuntimeError("Failed to find generated wheel package in distribution directory")
        whl_path = os.path.join(dist_dir, whl_files[0])

        # 临时解压目录
        temp_whl_dir = os.path.join(project_root, "temp_whl")
        os.makedirs(temp_whl_dir, exist_ok=True)

        # 解压whl
        with zipfile.ZipFile(whl_path, 'r') as zipf:
            zipf.extractall(temp_whl_dir)

        # 拷贝.so到解压后的目录
        native_so_dir = os.path.join(project_root, "native")  # 改用project_root，避免依赖外部HERE
        target_so_dir = os.path.join(temp_whl_dir, "vsparse/native")
        os.makedirs(target_so_dir, exist_ok=True)

        # Define target .so file prefixes to be included in the wheel package
        target_prefixes = ["_local_kvstore", "_offload_ops", "_paged_kmeans", "_prefetch_engine"]
        copied_count = 0

        # Iterate through native directory and copy matching .so files
        for file in os.listdir(native_so_dir):
            if any(file.startswith(p) for p in target_prefixes) and file.endswith(".so"):
                src_path = os.path.join(native_so_dir, file)
                dst_path = os.path.join(target_so_dir, file)
                shutil.copy2(src_path, dst_path)
                copied_count += 1

        # Validate that at least one .so file was copied
        if copied_count == 0:
            raise RuntimeError("No native shared library files (.so) found for integration")

        # 重新打包whl（覆盖原文件）
        os.remove(whl_path)
        with zipfile.ZipFile(whl_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_whl_dir):  # 用_替代未使用的dirs变量
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_whl_dir)
                    zipf.write(file_path, rel_path)

        # 清理临时目录
        shutil.rmtree(temp_whl_dir)


# 包路径与配置
current_script_dir = os.path.abspath(os.path.dirname(__file__))
native_cmake_dir = os.path.join(current_script_dir, "native")

# 查找子模块
packages = find_packages(
    where=".",
    include=[
        "bmsa",
        "bmsa.*",
        "connector",
        "connector.*",
        "connectors",
        "core",
        "kvstore",
        "native",
        "native.*",
        "store",
        "store.*",
    ],
)
packages = [f"vsparse.{pkg}" for pkg in packages]
packages.append("vsparse")

# 包路径映射
package_dir = {
    "vsparse": ".",
    "vsparse.bmsa": "bmsa",
    "vsparse.bmsa.prefetch": "bmsa/prefetch",
    "vsparse.bmsa.paged_kmeans": "bmsa/paged_kmeans",
    "vsparse.connector": "connector",
    "vsparse.connectors": "connectors",
    "vsparse.core": "core",
    "vsparse.kvstore": "kvstore",
    "vsparse.native": "native",
    "vsparse.native.kvstore": "native/kvstore",
    "vsparse.store": "store",
    "vsparse.store.localstore": "store/localstore",
}

# 扩展配置
cmake_ext = Extension("vsparse.native._native_build", sources=[])
cmake_ext.cmake_project = CMakeProject(cmake_lists_dir=native_cmake_dir, build_subdir="sparse_native")

# 最终setup配置（核心：注册自定义的bdist_wheel命令）
setup(
    name="vsparse",
    version="0.0.1",
    packages=packages,
    package_dir=package_dir,
    include_package_data=True,
    package_data={
        "": ["*.py", "*.so", "*.so.*", "*.pyd"],
        "vsparse.native": ["*.so", "*.so.*", "*.pyd"],
    },
    ext_modules=[cmake_ext],
    # 注册自定义命令：用我们的BdistWheelWithSO替换原生bdist_wheel
    cmdclass={
        "build_ext": CMakeBuildExt,
        "bdist_wheel": BdistWheelWithSO
    },
    python_requires=">=3.10,<3.14",
    zip_safe=False,
    # 指定whl输出目录
    options={"bdist_wheel": {"dist_dir": "dist"}}
)
