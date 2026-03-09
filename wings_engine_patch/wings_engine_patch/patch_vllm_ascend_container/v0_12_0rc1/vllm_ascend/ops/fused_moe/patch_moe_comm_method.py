# Copyright (c) 2025 Henan KunLun Technologies Co., Ltd. All rights reserved.

import wrapt
from typing import Optional, Any, Tuple
import torch

def patch_moe_comm_method_fused_experts():
    def apply_patch(module):
        # We need to patch MoECommMethod.fused_experts
        # Since MoECommMethod is a class, we need to patch the method on the class.
        
        # Original method signature:
        # fused_experts(self, hidden_states, w1, w2, topk_weights, topk_ids, activation="silu", 
        #               apply_router_weight_on_input=False, use_int8_w8a8=False, use_int4_w4a8=False, 
        #               use_fp8_w8a16=False, use_int4_w4a16=False, ...)
        
        # The prompt shows that `use_fp8_w8a16` was added to the arguments.
        # And the implementation uses it.
        
        # We will create a wrapper or replacement for fused_experts.
        # Since this is a method in a class (MoECommMethod), we can replace it.
        
        original_fused_experts = module.MoECommMethod.fused_experts

        def fused_experts_new(
            self,
            hidden_states: torch.Tensor,
            w1: Any, # torch.Tensor | list[torch.Tensor],
            w2: Any, # torch.Tensor | list[torch.Tensor],
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            activation: str = "silu",
            apply_router_weight_on_input: bool = False,
            use_int8_w8a8: bool = False,
            use_int4_w4a8: bool = False,
            use_fp8_w8a16: bool = False, # Added logic
            use_int4_w4a16: bool = False,
            global_num_experts: Optional[int] = None,
            expert_map: Optional[torch.Tensor] = None,
            w1_scale: Optional[Any] = None,# list[torch.Tensor]
            w2_scale: Optional[Any] = None,# list[torch.Tensor]
            w1_scale_bias: torch.Tensor = None,
            w2_scale_bias: torch.Tensor = None,
            w1_offset: Optional[torch.Tensor] = None,
            w2_offset: Optional[torch.Tensor] = None,
            # For Cube/Vector parallel
            shared_experts: Optional[Any] = None,
            quantized_x_for_share: Optional[Any] = None,
            dynamic_scale_for_share: Optional[Any] = None,
            # For load balance
            log2phy: torch.Tensor = None,
            global_redundant_expert_num: int = 0,
            need_trans: bool = False,
            dynamic_eplb: bool = False,
            mc2_mask: torch.Tensor = None,
            pertoken_scale: Optional[torch.Tensor] = None
        ):
            from vllm.forward_context import get_forward_context
            from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
            
    
            # Check constraints
            assert hidden_states.dtype in [
                torch.float32, torch.float16, torch.bfloat16, torch.int8
            ]
    
            moe_comm_method = get_forward_context().moe_comm_method
            assert moe_comm_method is not None, "Missing communication context"
    
            results = self.token_dispatcher.token_dispatch(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
                log2phy=log2phy,
                global_redundant_expert_num=global_redundant_expert_num,
                shared_experts=shared_experts,
                quantized_x_for_share=quantized_x_for_share,
                dynamic_scale_for_share=dynamic_scale_for_share,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                with_quant=use_int8_w8a8 or use_int4_w4a8, # or use_fp8_w8a16, soft fp8 don't do dynamic quant
                dynamic_eplb=dynamic_eplb,
                pertoken_scale=pertoken_scale)
    
            permuted_hidden_states, expert_tokens, dynamic_scale, group_list_type, topk_scales, context_metadata = \
                results["hidden_states"], results["group_list"], results.get("dynamic_scale"), results["group_list_type"], results.get("topk_scales"), results.get("context_metadata")
    
            mlp_output = unified_apply_mlp(hidden_states=permuted_hidden_states,
                                           w1=w1,
                                           w1_scale=w1_scale,
                                           w2=w2,
                                           w2_scale=w2_scale,
                                           group_list=expert_tokens,
                                           dynamic_scale=dynamic_scale,
                                           group_list_type=group_list_type,
                                           w1_scale_bias=w1_scale_bias,
                                           w2_scale_bias=w2_scale_bias,
                                           w1_offset=w1_offset,
                                           w2_offset=w2_offset,
                                           topk_scales=topk_scales,
                                           with_quant=use_int8_w8a8
                                           or use_int4_w4a8 or use_int4_w4a16 or use_fp8_w8a16,
                                           use_fp8_w8a16=use_fp8_w8a16,
                                           fusion=use_int8_w8a8,
                                           need_trans=need_trans,
                                           dynamic_eplb=dynamic_eplb)
    
            final_hidden_states = self.token_dispatcher.token_combine(
                hidden_states=mlp_output, context_metadata=context_metadata)
    
            if dynamic_eplb:
                return (final_hidden_states, group_list_type, expert_tokens)
    
            return final_hidden_states

        module.MoECommMethod.fused_experts = fused_experts_new

    wrapt.register_post_import_hook(
        apply_patch,
        'vllm_ascend.ops.fused_moe.moe_comm_method'
    )
