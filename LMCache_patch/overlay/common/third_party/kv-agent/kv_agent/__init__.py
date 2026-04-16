__all__ = [
    "init_config",
    "set_log_enabled",
    "set_mantissa_loss_level",
    "set_qat_instance_num",
    "set_kv_data_dir",
    "blocks_save_with_path",
    "blocks_load_with_path",
    "cpu_bind",
    "blocks_exists",
    "blocks_save",
    "blocks_load",
    "tfmr_blocks_save",
    "tfmr_blocks_load",
    "sgl_blocks_save",
    "sgl_blocks_load",
]


import logging
import glob
import ctypes
from ctypes import CDLL
from pathlib import Path

# 导入主模块
from ._C import *  # noqa

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preload_core_libraries():
    lib_dir = Path(__file__).parent / "lib"
    if not lib_dir.exists():
        logger.error("Library directory does not exist: %s", lib_dir)
        return False

    # 优先加载的库
    priority_libs = {
        "libusdm_drv_s.so": "Base driver library",
        "libqat_s.so": "QAT main library",
        "libqatzip.so": "Main compression library",
        "libqzip.so": "Auxiliary library"
    }

    loaded_libs = set()

    # 加载优先级库（基础依赖先加载）
    for lib, desc in priority_libs.items():
        lib_path = lib_dir / lib
        if not lib_path.is_file():
            logger.warning(
                "[Preload] Missing required library: %s (%s)", lib, desc)
            continue
        try:
            CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            logger.debug("[Preload] Loaded: %s (%s)", lib, desc)
            loaded_libs.add(lib)
        except Exception as e:
            logger.error("[Preload] Failed to load %s: %s", lib, str(e))
            return False

    # 加载其他库
    for so_path in glob.glob(str(lib_dir / "*.so*")):
        so_name = Path(so_path).name
        if so_name in loaded_libs:
            continue
        if not Path(so_path).is_file():
            continue
        try:
            CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            logger.debug("[AutoLoad] Loaded: %s", so_name)
        except Exception as e:
            logger.error("[AutoLoad] Failed to load %s: %s", so_name, str(e))

    return True


# 执行加载
if not preload_core_libraries():
    raise RuntimeError("Failed to preload required shared libraries.")
