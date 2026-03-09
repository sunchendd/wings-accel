# Copyright (c) 2025 Henan KunLun Technologies Co., Ltd. All rights reserved.

from typing import List, Optional, Tuple, Union

import torch

from ..ops import fp4_ops
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter, PerTensorScaleParameter


class AscendW4A16NVFP4LinearMethod:

    def __init__(self) -> None:
        self.group_size = 16

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
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        weight_loader = getattr(layer, 'weight_loader_v2',
                                extra_weight_attrs.get('weight_loader'))

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader)
        layer.register_parameter("weight_scale_2", weight_global_scale)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)
        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader)
        layer.register_parameter("input_scale", input_global_scale)

        layer.update_param_tp_status()

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        return fp4_ops.fp4_matmul(x, layer.weight.data, layer.weight_scale.data, layer.weight_scale_2.data)

    def process_weights_after_loading(self, layer):
        weight_data = fp4_ops.reformat_fp4_weight(layer.weight.data)
        weight_scale_data = layer.weight_scale.data.view(torch.uint8)
        weight_scale_2_data = layer.weight_scale_2.data.to(torch.bfloat16)
        del layer.input_scale

        # for torch compile
        layer.weight = torch.nn.Parameter(weight_data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            weight_scale_data, requires_grad=False)
        layer.weight_scale_2 = torch.nn.Parameter(
            weight_scale_2_data, requires_grad=False)
