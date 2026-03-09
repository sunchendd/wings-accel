# Copyright (c) 2025 Henan KunLun Technologies Co., Ltd. All rights reserved.
import torch
import torch_npu
import wrapt
from typing import Optional, List
from .. import fp8_ops

def quant_apply_mlp_new(hidden_states: torch.Tensor,
                    w1: list[torch.Tensor],
                    w1_scale: list[torch.Tensor],
                    w2: list[torch.Tensor],
                    w2_scale: list[torch.Tensor],
                    group_list: torch.Tensor,
                    group_list_type: int = 1,
                    dynamic_scale: torch.Tensor = None,
                    w1_scale_bias: torch.Tensor = None,
                    w2_scale_bias: torch.Tensor = None,
                    w1_offset: Optional[torch.Tensor] = None,
                    w2_offset: Optional[torch.Tensor] = None,
                    fusion: bool = False,
                    dynamic_eplb: bool = False,
                    use_fp8_w8a16: bool = False, # Added parameter
                    **kwargs # Capture unexpected args to be safe
                    ) -> torch.Tensor:
    from vllm_ascend.utils import dispose_tensor, enable_custom_op
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ascend_forward_context import MoECommType
    from vllm_ascend.ops.fused_moe.moe_mlp import _custom_gmm_swiglu_enabled, cumsum_group_list

    if use_fp8_w8a16:
        # gmm1: gate_up_proj
        hidden_states = fp8_ops.fp8_grouped_matmul(
            hidden_states,
            w1,
            w1_scale,
            group_list,
            group_list_type)
        # act_fn: swiglu
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        # gmm2: down_proj
        hidden_states = fp8_ops.fp8_grouped_matmul(
            hidden_states,
            w2,
            w2_scale,
            group_list,
            group_list_type)
        return hidden_states

    # Fallback to original logic
    if w1_offset is not None:
        unquantized_hidden_states = hidden_states
        quantized_hidden_states = None
    elif dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
            hidden_states)
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        unquantized_hidden_states = None
        pertoken_scale = dynamic_scale
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    _output_dtype = w2_scale[0].dtype

    weight_prefetch_method = get_forward_context().weight_prefetch_method
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(
            hidden_states)
    is_mc2 = get_forward_context().moe_comm_type == MoECommType.MC2

    if w1_scale_bias is None and w1_offset is None and is_mc2:
        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
             hidden_states, swiglu_out_scale, _ = (
                torch.ops._C_ascend.
                grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                    x=hidden_states,
                    weight=w1,
                    weight_scale=w1_scale,
                    x_scale=pertoken_scale,
                    group_list=cumsum_group_list(group_list, group_list_type,
                                                 0),
                ))
        elif fusion and not dynamic_eplb:
            hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1[0],
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=w1_scale[0],
                x_scale=pertoken_scale)
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            if w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                split_item=3,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32)[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=w1_scale[0],
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=cumsum_group_list(group_list, group_list_type, 1),
                activate_left=True,
                quant_mode=1,
            )
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=w2_scale[0].dtype)[0]
    elif w1_offset is not None:
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[unquantized_hidden_states],
            weight=[w1],
            antiquant_scale=[w1_scale],
            antiquant_offset=[w1_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
        dispose_tensor(unquantized_hidden_states)
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            antiquant_scale=[w2_scale],
            antiquant_offset=[w2_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
    else:
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat(
                    [group_list[:1],
                     torch.diff(group_list, dim=0)])
                group_list_type = 1
            bias1 = [w1_scale_bias] if not fusion else w1_scale_bias
            bias2 = [w2_scale_bias]
            _output_dtype = torch.bfloat16

        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
            hidden_states, swiglu_out_scale, _ = (
                torch.ops._C_ascend.
                grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                    x=hidden_states,
                    weight=w1,
                    weight_scale=w1_scale,
                    x_scale=pertoken_scale,
                    group_list=cumsum_group_list(group_list, group_list_type,
                                                 0),
                    bias=bias1,
                ))
        elif fusion and not dynamic_eplb:
            hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1[0],
                bias=bias1,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=w1_scale[0],
                x_scale=pertoken_scale)
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            w1_scale[0] = w1_scale[0].to(w2_scale[0].dtype)
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                scale=w1_scale,
                bias=bias1,
                per_token_scale=[pertoken_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=_output_dtype)[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            hidden_states = torch_npu.npu_swiglu(hidden_states)
            hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(
                hidden_states)
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
            bias=bias2,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
    return hidden_states


def unified_apply_mlp_new(hidden_states: torch.Tensor,
                      w1:  torch.Tensor | list[torch.Tensor],
                      w2:  torch.Tensor | list[torch.Tensor],
                      group_list: torch.Tensor,
                      w1_scale: Optional[list[torch.Tensor]] = None,
                      w2_scale: Optional[list[torch.Tensor]] = None,
                      dynamic_scale: torch.Tensor = None,
                      group_list_type: int = 1,
                      w1_scale_bias: torch.Tensor = None,
                      w2_scale_bias: torch.Tensor = None,
                      w1_offset: Optional[torch.Tensor] = None,
                      w2_offset: Optional[torch.Tensor] = None,
                      topk_scales: Optional[torch.Tensor] = None,
                      with_quant: bool = False,
                      fusion: bool = False,
                      need_trans: bool = True,
                      dynamic_eplb: bool = False,
                      use_fp8_w8a16: bool = False,
                      **kwargs) -> torch.Tensor:
    
    from vllm_ascend.ops.fused_moe.moe_mlp import unquant_apply_mlp

    if with_quant:
        assert w1_scale is not None and w2_scale is not None
        return quant_apply_mlp_new(hidden_states=hidden_states,
                               w1=w1,
                               w1_scale=w1_scale,
                               w2=w2,
                               w2_scale=w2_scale,
                               group_list=group_list,
                               dynamic_scale=dynamic_scale,
                               group_list_type=group_list_type,
                               w1_scale_bias=w1_scale_bias,
                               w2_scale_bias=w2_scale_bias,
                               w1_offset=w1_offset,
                               w2_offset=w2_offset,
                               fusion=fusion,
                               dynamic_eplb=dynamic_eplb,
                               use_fp8_w8a16=use_fp8_w8a16)
    else:
        return unquant_apply_mlp(hidden_states=hidden_states,
                                 w1=w1,
                                 w2=w2,
                                 group_list=group_list,
                                 group_list_type=group_list_type,
                                 topk_scales=topk_scales,
                                 need_trans=need_trans)

def patch_moe_mlp_functions():
    def apply_patch(module):
        module.quant_apply_mlp = quant_apply_mlp_new
        module.unified_apply_mlp = unified_apply_mlp_new

    wrapt.register_post_import_hook(
        apply_patch,
        'vllm_ascend.ops.fused_moe.moe_mlp'
    )
