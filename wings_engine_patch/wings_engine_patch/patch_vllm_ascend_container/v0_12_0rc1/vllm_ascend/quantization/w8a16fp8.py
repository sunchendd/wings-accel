from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import torch
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config

from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from ..ops import fp8_ops
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.fused_moe import (FusedMoeWeightScaleSupported)
from vllm.model_executor.parameter import (BlockQuantScaleParameter)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    validate_fp8_block_shape)

def ceil(x, y):
    return (x + y - 1) // y

class AscendW8A16FP8LinearMethod:

    def __init__(self):
        self.weight_block_size = (128, 128)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # adapted from vllm.model_executor.layers.quantization.fp8.FP8LinearMethod.create_weights
        output_size_per_partition = sum(output_partition_sizes)
        # When using FP8 quantization, all Linear classes in vllm prefer using weight_loader_v2 if available.
        weight_loader = getattr(layer, 'weight_loader_v2', None) or extra_weight_attrs.get('weight_loader')

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size

        validate_fp8_block_shape(layer, input_size, output_size,
                                    input_size_per_partition,
                                    output_partition_sizes,
                                    self.weight_block_size)

        weight = create_fp8_weight_parameter(output_size_per_partition,
                                                input_size_per_partition,
                                                weight_loader)
        layer.register_parameter("weight", weight)
        
        scale = create_fp8_scale_parameter(BlockQuantScaleParameter,
                                            output_partition_sizes,
                                            input_size_per_partition,
                                            self.weight_block_size,
                                            weight_loader)
        set_weight_attrs(scale, {"scale_type": "weight_scale"})
        layer.register_parameter("weight_scale_inv", scale)
        # set tp_rank & tp_size for weight and weight_scale_inv
        layer.update_param_tp_status()


    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        return fp8_ops.fp8_matmul(x, layer.weight.data, layer.weight_scale_inv.data)

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.view(torch.uint8)
        layer.weight_scale_inv.data = layer.weight_scale_inv.data.to(torch.float32)
        layer.weight.data, layer.weight_scale_inv.data = fp8_ops.optimize_weight_layout_for_fp8_matmul(
            layer.weight.data, layer.weight_scale_inv.data)
        
        # Replace with plain nn.Parameter to avoid torch.compile issues with custom Parameter classes
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(layer.weight_scale_inv.data, requires_grad=False)


class AscendW8A16FP8FusedMoEMethod:

    def __init__(self):
        self.transpose_weight = True

        self.ep_group = get_ep_group()

        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (vllm_config.compilation_config.mode
                             == CompilationMode.VLLM_COMPILE
                             and not vllm_config.model_config.enforce_eager)

        self.dynamic_eplb = ascend_config.dynamic_eplb or ascend_config.expert_map_record_path
        self.in_dtype = vllm_config.model_config.dtype

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = ""

    @staticmethod
    def create_weight_param_dict(num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(num_experts,
                                               2 *
                                               intermediate_size_per_partition,
                                               hidden_sizes,
                                               dtype=torch.float8_e4m3fn)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                              hidden_sizes,
                                              intermediate_size_per_partition,
                                              dtype=torch.float8_e4m3fn)
        return param_dict

    def create_dequant_scale_param_dict(self, num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale_inv"] = torch.empty(
                                                    num_experts,
                                                    2 * intermediate_size_per_partition // 128,
                                                    hidden_sizes // 128,
                                                    dtype=torch.float)
        param_dict["w2_weight_scale_inv"] = torch.empty(num_experts,
                                                    hidden_sizes // 128,
                                                    intermediate_size_per_partition // 128,
                                                    dtype=torch.float)
        return param_dict

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_param = self.create_weight_param_dict(
            num_experts, intermediate_size_per_partition, hidden_size)
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
        dequant_scale_param = self.create_dequant_scale_param_dict(
            num_experts, intermediate_size_per_partition, hidden_size)
        for param_key, param_value in dequant_scale_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
            1] == global_num_experts - global_redundant_expert_num, "Number of global experts mismatch (excluding redundancy)"

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts)

        # naive expert load balance for profiling
        if enable_force_load_balance:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        if self.use_aclgraph:
            moe_comm_method = get_forward_context().moe_comm_method
            return moe_comm_method.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight.data,
                w2=layer.w2_weight.data,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                use_fp8_w8a16=True,
                w1_scale=layer.w13_weight_scale_inv.data,
                w2_scale=layer.w2_weight_scale_inv.data,
                expert_map=expert_map,
                dynamic_eplb=self.dynamic_eplb,
                log2phy=log2phy,
                global_redundant_expert_num=global_redundant_expert_num)

        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        return moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight.data,
            w1_scale=layer.w13_weight_scale_inv.data,
            w2=layer.w2_weight.data,
            w2_scale=layer.w2_weight_scale_inv.data,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_fp8_w8a16=True,
            expert_map=expert_map,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            dynamic_eplb=self.dynamic_eplb)

    def process_weights_after_loading(self, layer):
        layer.w2_weight.data = layer.w2_weight.data.view(torch.uint8)
        layer.w13_weight.data = layer.w13_weight.data.view(torch.uint8)
        layer.w2_weight_scale_inv.data = layer.w2_weight_scale_inv.data.to(torch.float32)
        layer.w13_weight_scale_inv.data = layer.w13_weight_scale_inv.data.to(torch.float32)
        layer.w2_weight.data, layer.w2_weight_scale_inv.data = fp8_ops.optimize_weight_layout_for_fp8_grouped_matmul(
            layer.w2_weight.data, layer.w2_weight_scale_inv.data)
        layer.w13_weight.data, layer.w13_weight_scale_inv.data = fp8_ops.optimize_weight_layout_for_fp8_grouped_matmul(
            layer.w13_weight.data, layer.w13_weight_scale_inv.data)


# Placeholder for FP8 KV cache quantization to avoid errors during weight loading.
class AscendW8A16FP8KVCacheMethod:
    def create_weights(self, layer: torch.nn.Module) -> None:
        layer.k_scale = torch.nn.Parameter(
            torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(
            torch.tensor(-1.0), requires_grad=False)


