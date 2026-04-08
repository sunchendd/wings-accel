from typing import List


def enable(inference_engine: str, features: List[str], version: str):
    from .registry_v1 import enable as enable_v1
    return enable_v1(inference_engine, features, version)
