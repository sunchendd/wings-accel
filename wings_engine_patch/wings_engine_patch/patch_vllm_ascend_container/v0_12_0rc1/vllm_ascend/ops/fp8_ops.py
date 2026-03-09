import torch
import torch_npu


def _dequant_fp8_weight(w: torch.Tensor, s: torch.Tensor, g: torch.Tensor, group_list_type=0):
    assert w.is_contiguous() and s.is_contiguous() and g.is_contiguous()
    return torch_npu.npu_dequant_fp8_weight(w, s, g, group_list_type)


def fp8_matmul(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    '''
    x: [M, K], bf16
    w: [N, K], uint8 (actually fp8)
    s: [ceil(N / 128), ceil(K / 128)], fp32
    return: [M, N], bf16
    '''
    ACL_FORMAT_ND = 2
    assert torch_npu.get_npu_format(w) == ACL_FORMAT_ND and torch_npu.get_npu_format(s) == ACL_FORMAT_ND
    # TODO: support x of NCHW format
    if x.dim() == 2 and x.size(0) <= 128 and w.size(0) % 256 == 0 and w.size(1) % 256 == 0:
        return torch_npu.npu_soft_fp8_gemm(x, w.T, s.T)
    else:
        if w.is_contiguous() and s.is_contiguous():
            # param 'g' is not used in this case
            dequanted = _dequant_fp8_weight(w, s, torch.empty(
                1, dtype=torch.int64, device=x.device))
            return (x @ dequanted.T)
        else:
            w = w.T
            s = s.T
            # param 'g' is not used in this case
            dequanted = _dequant_fp8_weight(w, s, torch.empty(
                1, dtype=torch.int64, device=x.device))
            return (x @ dequanted)


def fp8_grouped_matmul(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, g: torch.Tensor, group_list_type) -> torch.Tensor:
    '''
    x: [M, K], bf16
    w: [G, N, K], uint8 (actually fp8)
    s: [G, N // 128, K // 128], fp32    note: the last two dims of s must be multiples of 128
    g: [G], int64
    return: [M, N], bf16
    '''
    if w.is_contiguous() and s.is_contiguous():
        dequanted = _dequant_fp8_weight(w, s, g, group_list_type=group_list_type)
        out = torch_npu.npu_grouped_matmul(
            [x], [dequanted.mT], group_list=g, group_list_type=group_list_type, group_type=0, split_item=2)
        return out[0]
    else:
        dequanted = _dequant_fp8_weight(w.mT, s.mT, g, group_list_type=group_list_type)
        out = torch_npu.npu_grouped_matmul(
            [x], [dequanted], group_list=g, group_list_type=group_list_type, group_type=0, split_item=2)
        return out[0]


def optimize_weight_layout_for_fp8_matmul(w: torch.Tensor, s: torch.Tensor):
    '''
    w: [N, K], uint8 (actually fp8)
    s: [ceil(N / 128), ceil(K / 128)], fp32
    '''
    assert w.is_contiguous() and s.is_contiguous()
    if w.size(0) >= 8192 and w.size(1) == 5120:
        w = w.T.contiguous().T
        s = s.T.contiguous().T
    return w, s


def optimize_weight_layout_for_fp8_grouped_matmul(w: torch.Tensor, s: torch.Tensor):
    '''
    w: [G, N, K], uint8 (actually fp8)
    s: [G, N // 128, K // 128], fp32    note: the last two dims of s must be multiples of 128
    '''
    assert w.is_contiguous() and s.is_contiguous()
    if w.size(1) == 2048:
        w = w.mT.contiguous().mT
        s = s.mT.contiguous().mT
    return w, s
