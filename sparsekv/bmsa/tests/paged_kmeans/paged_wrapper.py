import torch


def extract_contiguous_kv_from_paged(
    kv_cache,
    block_table,
    start_pos,
    seq_len,
):
    """
    从 paged KVCache 中抽取连续 KV
    输出:
      key, value: [H, seq_len, D]
    """
    _, _, block_size, H, D = kv_cache.shape
    key = torch.empty((H, seq_len, D), device=kv_cache.device, dtype=kv_cache.dtype)
    value = torch.empty_like(key)

    for t in range(seq_len):
        pos = start_pos + t
        logical_block = pos // block_size
        offset = pos % block_size
        phys = block_table[logical_block]

        key[:, t] = kv_cache[0, phys, offset]
        value[:, t] = kv_cache[1, phys, offset]

    return key, value
