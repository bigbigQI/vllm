import logging
from typing import Callable, Optional, Union
from unittest.mock import patch

import torch
from torch.nn import Parameter

logger = logging.getLogger(__name__)


def _create_param_from_subclass_attributes(custom_data: torch.Tensor,
                                           custom_weight) -> Parameter:
    """
    Helper to preserve custom attributes from ModelWeightParameter and
    PerTensorScaleParameter when creating new Parameters.
    """
    param = Parameter(custom_data, requires_grad=False)
    base_param_dir = dir(torch.nn.Parameter)
    custom_weight_dir = dir(custom_weight)
    # Find the attributes that are unique to the custom parameter
    custom_attributes = [
        attr for attr in custom_weight_dir
        if attr not in base_param_dir and not attr.startswith("__")
    ]
    # Set the custom attributes into the base parameter object
    for attr in custom_attributes:
        setattr(param, attr, getattr(custom_weight, attr))
    return param


def process_weights_after_loading_modelopt(self,
                                           layer: torch.nn.Module) -> None:
    """
    Patched process_weights_after_loading for ModelOptNvFp4LinearMethod.
    Key differences from original:
    1. Preserves tensor attributes using _create_param_from_subclass_attributes
    2. Creates separate marlin_* weights instead of overwriting originals
    3. Handles weight_scale_2_max separately for Marlin
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        swizzle_blockscale)
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_bias,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        mxfp4_marlin_process_scales,
        nvfp4_marlin_process_scales,
        nvfp4_marlin_process_global_scale,
    )

    def prepare_fp4_layer_for_marlin(layer: torch.nn.Module,
                                     weight_scale_2_max: torch.Tensor) -> None:
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression will "
            "be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )

        is_nvfp4 = hasattr(layer, "weight_scale_2")
        group_size = 16 if is_nvfp4 else 32

        part_size_n = layer.output_size_per_partition
        part_size_k = layer.input_size_per_partition
        param_dtype = layer.params_dtype

        assert layer.weight.shape == (part_size_n, part_size_k // 2)

        device = layer.weight.device

        # WORKSPACE
        if getattr(layer, "workspace", None) is None:
            layer.workspace = marlin_make_workspace_new(device)

        # WEIGHT
        # Repack weights to marlin format
        perm = torch.empty(0, dtype=torch.int, device=device)
        qweight = layer.weight.view(torch.int32).T.contiguous()

        marlin_qweight = ops.gptq_marlin_repack(
            b_q_weight=qweight,
            perm=perm,
            size_k=part_size_k,
            size_n=part_size_n,
            num_bits=4,
        )
        layer.marlin_weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

        # WEIGHT SCALES
        # Permute scales
        weight_scale = layer.weight_scale.T.contiguous()

        if not is_nvfp4:
            weight_scale = weight_scale.view(torch.float8_e8m0fnu)

        weight_scale = weight_scale.to(param_dtype)
        weight_scale = marlin_permute_scales(
            s=weight_scale, size_k=part_size_k, size_n=part_size_n, group_size=group_size
        )

        if is_nvfp4:
            weight_scale = nvfp4_marlin_process_scales(weight_scale)
            layer.marlin_weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

            weight_scale_2 = weight_scale_2_max.to(param_dtype)
            weight_scale_2 = nvfp4_marlin_process_global_scale(weight_scale_2)
            layer.marlin_weight_scale_2 = torch.nn.Parameter(weight_scale_2, requires_grad=False)
        else:
            weight_scale = mxfp4_marlin_process_scales(weight_scale)
            layer.marlin_weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        if hasattr(layer, "bias") and layer.bias is not None:
            assert layer.bias.shape == (part_size_n,)
            bias = marlin_permute_bias(layer.bias)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        return

    # global scales:
    input_scale_2 = layer.input_scale.data
    layer.input_scale = _create_param_from_subclass_attributes(input_scale_2, layer.input_scale)
    input_scale_2_max = input_scale_2.max().to(torch.float32)

    weight_scale_2 = layer.weight_scale_2.data
    layer.weight_scale_2 = _create_param_from_subclass_attributes(weight_scale_2, layer.weight_scale_2)
    weight_scale_2_max = weight_scale_2.max().to(torch.float32)

    layer.alpha = Parameter(input_scale_2_max * weight_scale_2_max,
                            requires_grad=False)

    # Calculate `1 / input_scale` so that we don't need to do so at runtime
    layer.input_scale_inv = Parameter(
        (1 / layer.input_scale).to(torch.float32), requires_grad=False)

    # Swizzle the weight blockscale.
    # contracting dimension is input dimension
    # block_size = 16;
    assert (layer.weight_scale.dtype == torch.float8_e4m3fn), (
        "Weight Block scale must be represented as FP8-E4M3")

    if self.backend == "marlin":
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        layer.weight = _create_param_from_subclass_attributes(weight, layer.weight)
        layer.weight_scale = _create_param_from_subclass_attributes(weight_scale, layer.weight_scale)
        prepare_fp4_layer_for_marlin(layer, weight_scale_2_max)

        if getattr(layer, "prefix", None) == "model.layers.27.mlp.gate_up_proj" or getattr(layer, "prefix", "").startswith("model.layers.27.self_attn"):
            print(f"##VLLM-MARLIN##: {getattr(layer, 'prefix', None)}: {layer.marlin_weight.data[0, :4]}, scale: {layer.marlin_weight_scale.data[0, :4]}, scale_2: {layer.marlin_weight_scale_2.data}")

        del layer.alpha
        # del layer.input_scale
    elif self.backend == "flashinfer-trtllm":
        # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
        # FlashInfer provides nvfp4_quantize to quantize + shuffle the
        # layout but we use our own quantization so we have to call
        # shuffles ourselves.
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        weight = layer.weight.data
        weight_scale = layer.weight_scale.data

        epilogue_tile_m = 128
        weight = shuffle_matrix_a(weight.view(torch.uint8),
                                    epilogue_tile_m)
        weight_scale = (shuffle_matrix_sf_a(weight_scale.view(
            torch.uint8), epilogue_tile_m).reshape(
                weight_scale.shape).view(torch.float8_e4m3fn))

        layer.weight_scale = _create_param_from_subclass_attributes(weight_scale, layer.weight_scale)
        layer.weight = _create_param_from_subclass_attributes(weight, layer.weight)
    else:
        swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
        layer.weight_scale = _create_param_from_subclass_attributes(swizzled_weight_scale, layer.weight_scale)
        layer.weight = _create_param_from_subclass_attributes(layer.weight.data, layer.weight)


def apply_modelopt(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
    from vllm.utils.flashinfer import (flashinfer_scaled_fp4_mm)
    if self.backend == "marlin":
        # if getattr(layer, "prefix", None) == "model.layers.27.mlp.gate_up_proj" or getattr(layer, "prefix", "").startswith("model.layers.27.self_attn"):
            # print(f"##VLLM-MARLIN##: {getattr(layer, 'prefix', None)}: {layer.marlin_weight.data[0, :4]}, scale: {layer.marlin_weight_scale.data[0, :4]}, scale_2: {layer.marlin_weight_scale_2.data}")
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.marlin_weight,
            weight_scale=layer.marlin_weight_scale,
            weight_scale_2=layer.marlin_weight_scale_2,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias)

    output_dtype = x.dtype
    output_shape = [x.shape[0], layer.weight.shape[0]]

    # quantize BF16 or FP16 to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = scaled_fp4_quant(x, layer.input_scale_inv)

    # validate dtypes of quantized input, input block scale,
    # weight and weight_blockscale
    assert (x_fp4.dtype == torch.uint8)
    assert (layer.weight.dtype == torch.uint8)
    assert (x_blockscale.dtype == torch.float8_e4m3fn)
    assert (layer.weight_scale.dtype == torch.float8_e4m3fn)
    assert (layer.alpha.dtype == torch.float32)

    mm_args = (
        x_fp4,
        layer.weight,
        x_blockscale,
        layer.weight_scale,
        layer.alpha,
        output_dtype,
    )
    if self.backend == "flashinfer-trtllm":
        out = flashinfer_scaled_fp4_mm(*mm_args, backend="trtllm")
    elif self.backend == "flashinfer-cutlass":
        out = flashinfer_scaled_fp4_mm(*mm_args, backend="cutlass")
    else:
        out = cutlass_scaled_fp4_mm(*mm_args)

    if bias is not None:
        out = out + bias
    return out.view(*output_shape)

def process_weights_after_loading_kv(self, layer: torch.nn.Module) -> None:
    """Patched KV cache method - placeholder for static scales."""
    # Import the original logic
    from vllm.model_executor.layers.quantization.kv_cache import (
        BaseKVCacheMethod)

    # Preserve k_scale and v_scale if they exist
    if hasattr(layer, "k_scale") and layer.k_scale is not None:
        layer.k_scale = Parameter(layer.k_scale.data.max(),
                                  requires_grad=False)
    if hasattr(layer, "v_scale") and layer.v_scale is not None:
        layer.v_scale = Parameter(layer.v_scale.data.max(),
                                  requires_grad=False)


# =============================================================================
# ModelOptNvFp4FusedMoE Patches
# =============================================================================


def process_weights_after_loading_moe(self, layer: torch.nn.Module) -> None:
    """
    Patched process_weights_after_loading for ModelOptNvFp4FusedMoE.
    Key differences from original:
    1. Preserves tensor attributes using _create_param_from_subclass_attributes
    2. Creates separate marlin_* weights for MoE instead of overwriting originals
    3. Handles weight_scale_2 separately for Marlin
    """
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_bias,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        swizzle_blockscale)
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        reorder_w1w3_to_w3w1)

    def prepare_moe_fp4_layer_for_marlin(
        layer: torch.nn.Module,
        w13_weight_scale_2_max: torch.Tensor,
        w2_weight_scale_2_max: torch.Tensor,
    ) -> None:
        """
        Prepare MoE FP4 layer for Marlin kernel, creating separate marlin_*
        weights instead of overwriting original weights.
        """
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression will "
            "be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads.")

        group_size = 16  # NVFP4 uses group_size=16

        e = layer.num_experts
        k = layer.hidden_size
        n = layer.intermediate_size_per_partition

        # WORKSPACE
        device = layer.w13_weight.device
        param_dtype = layer.params_dtype
        if getattr(layer, "workspace", None) is None:
            layer.workspace = marlin_make_workspace_new(device, 4)
        perm = torch.empty(0, dtype=torch.int, device=device)

        # WEIGHT - Repack weights to marlin format
        for name in ["w13_weight", "w2_weight"]:
            weight = getattr(layer, name)
            tensor_list = []
            if "w13" in name:
                size_n, size_k = n * 2, k
            else:
                size_n, size_k = k, n

            assert weight.shape == (e, size_n, size_k // 2), (
                f"Expected {name} shape ({e}, {size_n}, {size_k // 2}), "
                f"got {weight.shape}")

            for i in range(e):
                qweight = weight[i].view(torch.int32).T.contiguous()

                marlin_qweight = ops.gptq_marlin_repack(
                    b_q_weight=qweight,
                    perm=perm,
                    size_k=size_k,
                    size_n=size_n,
                    num_bits=4)
                tensor_list.append(marlin_qweight)

            marlin_weight = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
            marlin_weight = torch.nn.Parameter(marlin_weight,
                                               requires_grad=False)

            # Store as marlin_* instead of overwriting original
            setattr(layer, f"marlin_{name}", marlin_weight)

        # WEIGHT SCALES - Permute scales
        for name in ["w13", "w2"]:
            scales = getattr(layer, name + "_weight_scale")
            scales = scales.to(param_dtype)

            if name == "w13":
                global_scale_max = w13_weight_scale_2_max
            else:
                global_scale_max = w2_weight_scale_2_max

            tensor_list = []
            if "w13" in name:
                size_n, size_k = n * 2, k
            else:
                size_n, size_k = k, n

            for i in range(e):
                scale = scales[i].T

                marlin_scales = marlin_permute_scales(
                    s=scale,
                    size_k=size_k,
                    size_n=size_n,
                    group_size=group_size)
                marlin_scales = nvfp4_marlin_process_scales(marlin_scales)
                tensor_list.append(marlin_scales)

            marlin_scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
            marlin_scales = torch.nn.Parameter(marlin_scales,
                                               requires_grad=False)
            setattr(layer, f"marlin_{name}_weight_scale", marlin_scales)

            # Process global scale
            global_scale = nvfp4_marlin_process_global_scale(
                global_scale_max.to(param_dtype))
            global_scale = torch.nn.Parameter(global_scale, requires_grad=False)
            setattr(layer, f"marlin_{name}_weight_scale_2", global_scale)

    # ==== Main processing logic ====

    # GEMM 1 processing - preserve attributes
    gemm1_weight = layer.w13_weight.data
    gemm1_weight_scale = layer.w13_weight_scale.data

    if self.allow_flashinfer and not self.use_marlin:
        gemm1_weight, gemm1_weight_scale = reorder_w1w3_to_w3w1(
            gemm1_weight, gemm1_weight_scale, dim=-2)

    layer.w13_weight = _create_param_from_subclass_attributes(
        gemm1_weight, layer.w13_weight)
    layer.w13_weight_scale = _create_param_from_subclass_attributes(
        gemm1_weight_scale, layer.w13_weight_scale)

    # Common processing for w13_weight_scale_2
    if not torch.allclose(layer.w13_weight_scale_2[:, 0],
                          layer.w13_weight_scale_2[:, 1]):
        logger.warning_once(
            "w1_weight_scale_2 must match w3_weight_scale_2. "
            "Accuracy may be affected.")

    w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
    w13_weight_scale_2_data = layer.w13_weight_scale_2.data
    layer.w13_weight_scale_2 = _create_param_from_subclass_attributes(
        w13_weight_scale_2, layer.w13_weight_scale_2)

    # Compute max for Marlin
    w13_weight_scale_2_max = w13_weight_scale_2.max().to(torch.float32)

    # Common processing for input scales and alphas
    w13_input_scale = layer.w13_input_scale.max(dim=1).values.to(torch.float32)
    layer.g1_alphas = Parameter(
        (w13_input_scale * w13_weight_scale_2).to(torch.float32),
        requires_grad=False)

    # This is for quantization, so we need to invert it.
    layer.w13_input_scale_quant = Parameter(
        (1 / w13_input_scale).to(torch.float32), requires_grad=False)

    # GEMM 2 processing
    w2_weight_scale_2_max = layer.w2_weight_scale_2.max().to(torch.float32)

    layer.g2_alphas = Parameter(
        (layer.w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
        requires_grad=False)

    # This is for quantization, so we need to invert it.
    layer.w2_input_scale_quant = Parameter(
        (1 / layer.w2_input_scale).to(torch.float32), requires_grad=False)

    # Backend-specific processing
    if self.use_marlin:
        # Prepare Marlin weights (creates marlin_* attributes)
        prepare_moe_fp4_layer_for_marlin(
            layer, w13_weight_scale_2_max, w2_weight_scale_2_max)

        # Clean up non-Marlin attributes
        del layer.g1_alphas
        del layer.g2_alphas
        del layer.w13_input_scale_quant
        del layer.w2_input_scale_quant

    elif (self.allow_flashinfer and
          self.flashinfer_moe_backend is not None):
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            FlashinferMoeBackend)

        if self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
            # TensorRT-LLM specific processing
            (gemm1_weights_fp4_shuffled, gemm1_scales_fp4_shuffled,
             gemm2_weights_fp4_shuffled, gemm2_scales_fp4_shuffled
             ) = self.prepare_static_weight_layouts_for_trtllm_moe(
                 layer.w13_weight,
                 layer.w2_weight,
                 layer.w13_weight_scale,
                 layer.w2_weight_scale,
                 layer.w2_weight.size(-2),  # hidden_size
                 layer.w13_weight.size(-2) // 2,  # intermediate_size
                 layer.w13_weight.size(0),  # num_experts
             )

            layer.gemm1_weights_fp4_shuffled = Parameter(
                gemm1_weights_fp4_shuffled, requires_grad=False)
            layer.gemm2_weights_fp4_shuffled = Parameter(
                gemm2_weights_fp4_shuffled, requires_grad=False)
            layer.gemm1_scales_fp4_shuffled = Parameter(
                gemm1_scales_fp4_shuffled, requires_grad=False)
            layer.gemm2_scales_fp4_shuffled = Parameter(
                gemm2_scales_fp4_shuffled, requires_grad=False)

            # Additional parameter needed for TRT-LLM
            layer.g1_scale_c = Parameter(
                (layer.w2_input_scale_quant * layer.g1_alphas).to(
                    torch.float32),
                requires_grad=False,
            )

            # Clean up weights that won't be used by TRT-LLM
            del layer.w2_weight
            del layer.w2_weight_scale
            del layer.w13_weight
            del layer.w13_weight_scale
        else:
            # CUTLASS backend
            assert (layer.w13_weight_scale.shape[2] % 16 == 0), (
                "Expected weight_scale.dim(1) to be divisible by 16")
            assert (layer.w13_weight_scale.dtype == torch.float8_e4m3fn), (
                "Weight Blockscale must be represented as FP8-E4M3")
            w13_blockscale_swizzled = swizzle_blockscale(layer.w13_weight_scale)
            layer.w13_weight_scale = _create_param_from_subclass_attributes(
                w13_blockscale_swizzled, layer.w13_weight_scale)

            assert (layer.w2_weight_scale.shape[2] % 16 == 0), (
                "Expected weight_scale.dim(1) to be divisible by 16")
            assert (layer.w2_weight_scale.dtype == torch.float8_e4m3fn), (
                "Weight Blockscale must be represented as FP8-E4M3")
            w2_blockscale_swizzled = swizzle_blockscale(layer.w2_weight_scale)
            layer.w2_weight_scale = _create_param_from_subclass_attributes(
                w2_blockscale_swizzled, layer.w2_weight_scale)
            layer.w2_weight = _create_param_from_subclass_attributes(
                layer.w2_weight.data, layer.w2_weight)
    else:
        # Default CUTLASS processing
        assert (layer.w13_weight_scale.shape[2] % 16 == 0), (
            "Expected weight_scale.dim(1) to be divisible by 16")
        assert (layer.w13_weight_scale.dtype == torch.float8_e4m3fn), (
            "Weight Blockscale must be represented as FP8-E4M3")
        w13_blockscale_swizzled = swizzle_blockscale(layer.w13_weight_scale)
        layer.w13_weight_scale = _create_param_from_subclass_attributes(
            w13_blockscale_swizzled, layer.w13_weight_scale)

        assert (layer.w2_weight_scale.shape[2] % 16 == 0), (
            "Expected weight_scale.dim(1) to be divisible by 16")
        assert (layer.w2_weight_scale.dtype == torch.float8_e4m3fn), (
            "Weight Blockscale must be represented as FP8-E4M3")
        w2_blockscale_swizzled = swizzle_blockscale(layer.w2_weight_scale)
        layer.w2_weight_scale = _create_param_from_subclass_attributes(
            w2_blockscale_swizzled, layer.w2_weight_scale)
        layer.w2_weight = _create_param_from_subclass_attributes(
            layer.w2_weight.data, layer.w2_weight)


def apply_moe(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    use_grouped_topk: bool = False,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    expert_load_view: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
    logical_replica_count: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Patched apply for ModelOptNvFp4FusedMoE.
    Uses marlin_* weights for Marlin backend.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.scalar_type import scalar_types

    if enable_eplb:
        raise NotImplementedError(
            "EPLB not supported for `ModelOptNvFp4FusedMoE` yet.")
    assert activation == "silu", "Only SiLU activation is supported."

    # TensorRT-LLM path
    if (self.allow_flashinfer and self.flashinfer_moe_backend is not None):
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            FlashinferMoeBackend)

        if self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
            import flashinfer

            from vllm.model_executor.models.llama4 import Llama4MoE
            from vllm.model_executor.layers.quantization.modelopt import (
                _get_tile_tokens_dim)

            assert self.fused_experts is None

            a1_gscale = layer.w13_input_scale_quant
            (hidden_states_fp4,
             hidden_states_scale_linear_fp4) = flashinfer.fp4_quantize(
                 x,
                 a1_gscale,
                 is_sf_swizzled_layout=False,
             )
            use_llama4_routing = (
                custom_routing_function is Llama4MoE.custom_routing_function)
            routing_method_type = flashinfer.RoutingMethodType.DeepSeekV3
            if use_llama4_routing:
                routing_method_type = flashinfer.RoutingMethodType.Llama4
            routing_bias = e_score_correction_bias
            if routing_bias is not None:
                routing_bias = routing_bias.to(torch.bfloat16)
            out = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
                routing_logits=(router_logits if use_llama4_routing
                                else router_logits.to(torch.float32)),
                routing_bias=routing_bias,
                hidden_states=hidden_states_fp4,
                hidden_states_scale=hidden_states_scale_linear_fp4.view(
                    torch.float8_e4m3fn).flatten(),
                gemm1_weights=layer.gemm1_weights_fp4_shuffled.data,
                gemm1_weights_scale=layer.gemm1_scales_fp4_shuffled.data.view(
                    torch.float8_e4m3fn),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=layer.gemm2_weights_fp4_shuffled.data,
                gemm2_weights_scale=layer.gemm2_scales_fp4_shuffled.data.view(
                    torch.float8_e4m3fn),
                gemm2_bias=None,
                output1_scale_scalar=layer.g1_scale_c.data,
                output1_scale_gate_scalar=layer.g1_alphas.data,
                output2_scale_scalar=layer.g2_alphas.data,
                num_experts=global_num_experts,
                top_k=top_k,
                n_group=(num_expert_group
                         if num_expert_group is not None else 0),
                topk_group=topk_group if topk_group is not None else 0,
                intermediate_size=layer.intermediate_size_per_partition,
                local_expert_offset=layer.ep_rank * layer.local_num_experts,
                local_num_experts=layer.local_num_experts,
                routed_scaling_factor=None,
                tile_tokens_dim=_get_tile_tokens_dim(
                    x.shape[0], top_k, layer.local_num_experts),
                routing_method_type=routing_method_type,
                do_finalize=True,
            )[0]
            return out

    # Expert selection
    topk_weights, topk_ids, _ = FusedMoE.select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        indices_type=self.topk_indices_dtype)

    # Marlin path - uses marlin_* weights
    if self.use_marlin:
        assert self.fused_experts is None
        return torch.ops.vllm.fused_marlin_moe(
            x,
            layer.marlin_w13_weight,
            layer.marlin_w2_weight,
            None,  # w1
            None,  # w2
            layer.marlin_w13_weight_scale,
            layer.marlin_w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            global_scale1=layer.marlin_w13_weight_scale_2,
            global_scale2=layer.marlin_w2_weight_scale_2,
            quant_type_id=scalar_types.float4_e2m1f.id,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            workspace=layer.workspace)

    # Modular kernel path
    if self.fused_experts is not None:
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            FlashinferMoeBackend)
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            is_valid_flashinfer_cutlass_fused_moe)

        assert (self.allow_flashinfer and
                self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS)

        assert is_valid_flashinfer_cutlass_fused_moe(
            x, layer.w13_weight, layer.w2_weight), (
                "Flashinfer CUTLASS Fused MoE not applicable!")

        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

    # FlashInfer CUTLASS path
    if (self.allow_flashinfer and self.flashinfer_moe_backend is not None):
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            FlashinferMoeBackend)

        if self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS:
            from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
                flashinfer_cutlass_moe_fp4)
            assert self.moe_quant_config is not None

            return flashinfer_cutlass_moe_fp4(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                quant_config=self.moe_quant_config,
                inplace=False,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

    # Default CUTLASS path
    from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4
    assert self.moe_quant_config is not None
    return cutlass_moe_fp4(
        a=x,
        w1_fp4=layer.w13_weight,
        w2_fp4=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_config=self.moe_quant_config,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
        m=x.shape[0],
        n=layer.w2_weight.shape[2] * 2,
        k=x.shape[1],
        e=layer.w13_weight.shape[0],
    )


def apply_vllm_modelopt_patches():
    """Apply all ModelOpt patches."""
    # Patch ModelOptNvFp4LinearMethod
    func1_path = ("vllm.model_executor.layers.quantization.modelopt."
                  "ModelOptNvFp4LinearMethod.process_weights_after_loading")
    patcher1 = patch(func1_path, process_weights_after_loading_modelopt)
    patcher1.start()

    func2_path = ("vllm.model_executor.layers.quantization.modelopt."
                  "ModelOptNvFp4LinearMethod.apply")
    patcher2 = patch(func2_path, apply_modelopt)
    patcher2.start()

    # Patch ModelOptNvFp4FusedMoE
    func3_path = ("vllm.model_executor.layers.quantization.modelopt."
                  "ModelOptNvFp4FusedMoE.process_weights_after_loading")
    patcher3 = patch(func3_path, process_weights_after_loading_moe)
    patcher3.start()

    func4_path = ("vllm.model_executor.layers.quantization.modelopt."
                  "ModelOptNvFp4FusedMoE.apply")
    patcher4 = patch(func4_path, apply_moe)
    patcher4.start()

    # Patch KV cache method for static scales
    func5_path = ("vllm.model_executor.layers.quantization.kv_cache."
                  "BaseKVCacheMethod.process_weights_after_loading")
    patcher5 = patch(func5_path, process_weights_after_loading_kv)
    patcher5.start()

    logger.info("Applied vLLM ModelOpt patches for Linear, MoE, and KV cache")