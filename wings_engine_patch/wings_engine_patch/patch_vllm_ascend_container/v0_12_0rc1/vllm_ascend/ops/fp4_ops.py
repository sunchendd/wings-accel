# Copyright (c) 2025 Henan KunLun Technologies Co., Ltd. All rights reserved.

import torch
import torch_npu


def _fp4_to_fp16(fp4_data: torch.Tensor):
    device = fp4_data.device
    qstate = torch.Tensor([
        0.0, 0.5, 1, 1.5, 2, 3, 4, 6,
        -0.0, -0.5, -1, -1.5, -2, -3, -4, -6
    ]).to(torch.float16).to(device)
    m, n = fp4_data.shape
    d = fp4_data.view(-1)
    unpacked = torch.empty(d.size(0) * 2, dtype=torch.uint8, device=device)
    unpacked[1::2] = d >> 4
    unpacked[0::2] = d & 0x0F
    return qstate.gather(0, unpacked.to(torch.int64)).view(m, n * 2)


def _fp8_to_fp16(uint8):
    device = uint8.device
    half = uint8.view(torch.int8).to(torch.float16)
    int16 = half.to(torch.int16)
    a = int16 << 7
    b = torch.tensor(0XBF80, dtype=torch.uint16).to(device).view(torch.int16)
    c = a & b
    d = torch.tensor(0x5c00, dtype=torch.int16).to(device).view(torch.int16)
    return c.view(torch.float16) * d.view(torch.float16)


def _apply_local_scale(weight, scale):
    assert weight.is_contiguous() and scale.is_contiguous()
    return torch_npu.npu_fp4_apply_local_scale(weight, scale)


def _soft_fp4_matmul(x, w, s, s2):
    bf16_weight = _apply_local_scale(w, s)
    mmres = (x @ bf16_weight.T)
    res = mmres * s2
    return res.to(torch.bfloat16)


def _merged_soft_fp4_matmul(x, w, s, s2):
    bf16_weight = _apply_local_scale(w, s)
    mmres = (x @ bf16_weight.T)
    d = mmres.size(1) // 2
    mmres[:, :d] *= s2[0]
    mmres[:, d:] *= s2[1]
    return mmres.to(torch.bfloat16)


def _qkv_merged_soft_fp4_matmul(x, w, s, s2):
    bf16_weight = _apply_local_scale(w, s)
    mmres = (x @ bf16_weight.T)
    # FIXME: only support Qwen3-32B-FP4 model now
    d = mmres.size(1) // 10
    mmres[:, :d * 8] *= s2[0]
    mmres[:, d * 8:d * 9] *= s2[1]
    mmres[:, d * 9:] *= s2[2]
    return mmres.to(torch.bfloat16)


def _reformat_fp4_weight_slice(weight: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.uint8
    device = weight.device
    orig_shape = weight.shape
    flat_weight = weight.view(-1)
    logic_len = orig_shape.numel() * 2

    low_bits = flat_weight & 0x0f
    high_bits = (flat_weight >> 4) & 0x0f
    unpacked = torch.empty(logic_len, dtype=torch.uint8, device=device)
    unpacked[::2] = low_bits
    unpacked[1::2] = high_bits

    low_bits = unpacked[:logic_len//2]
    high_bits = unpacked[logic_len // 2:]
    result = low_bits | (high_bits << 4)
    return result.view(orig_shape)


BASIC_W_LEN = 512
BASIC_SCALE_LEN = BASIC_W_LEN // 16
NUM_BASIC_PER_LOOP = 32


def reformat_fp4_weight(weight: torch.Tensor, uint8_len=NUM_BASIC_PER_LOOP * BASIC_W_LEN // 2) -> torch.Tensor:
    assert weight.dtype == torch.uint8
    assert weight.numel() % uint8_len == 0
    shape = weight.shape
    weight = weight.flatten()
    result = torch.empty_like(weight)
    for i in range(weight.numel() // uint8_len):
        s = slice(i*uint8_len, (i+1)*uint8_len)
        result[s] = _reformat_fp4_weight_slice(weight[s])
    return result.view(shape)


def fp4_matmul(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor) -> torch.Tensor:
    '''
    x: bf16, [batch_size, input_size]
    weight: uint8, [output_size, input_size / 2]
    weight_scale: uint8, [output_size, input_size / group_size], group_size=16
    weight_scale_2: float32, [1] or [2] or [3]
    return: bf16, [batch_size, output_size]
    '''
    assert weight_scale_2.size(0) in (1, 2, 3)
    if weight_scale_2.size(0) == 1:
        return _soft_fp4_matmul(x, weight, weight_scale, weight_scale_2)
    elif weight_scale_2.size(0) == 2:
        return _merged_soft_fp4_matmul(x, weight, weight_scale, weight_scale_2)
    elif weight_scale_2.size(0) == 3:
        return _qkv_merged_soft_fp4_matmul(x, weight, weight_scale, weight_scale_2)
    else:
        raise ValueError("Unsupported weight_scale_2 size for fp4_matmul.")
