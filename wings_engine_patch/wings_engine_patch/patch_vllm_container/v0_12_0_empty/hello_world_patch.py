import sys


def patch_vllm_hello_world():
    print('[Vllm Patch] hello_world patch enabled', file=sys.stderr)
