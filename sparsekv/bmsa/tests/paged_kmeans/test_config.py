from vllm.engine.arg_utils import EngineArgs
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vsparse.bmsa.paged_kmeans.config import build_default_kmeans_config


def test_build_default_kmeans_config_uses_cache_dtype_field():
    engine_args = EngineArgs(
        model="/home/models/Qwen3-0.6B",
        max_model_len=128,
        enforce_eager=True,
        gpu_memory_utilization=0.1,
        sparse_config={
            "enable_sparse": True,
            "sparse_algo_type": "CRSA",
        },
    )
    vllm_config = engine_args.create_engine_config()

    cfg = build_default_kmeans_config(vllm_config)

    expected_dtype = kv_cache_dtype_str_to_dtype(
        vllm_config.cache_config.cache_dtype, vllm_config.model_config
    )
    assert cfg.dtype == expected_dtype
    assert cfg.max_num_centroids < 8192
