def process_weights_after_loading(self, layer: Module) -> None:
    from vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod import _swap_w13_to_w31, _is_col_major
    from vllm.utils.deep_gemm import is_blackwell_deep_gemm_used
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_col_major_tma_aligned_tensor, requant_weight_ue8m0_inplace)
    assert self.quant_config.activation_scheme == "dynamic"
    if self.flashinfer_moe_enabled:
        w13_weight = _swap_w13_to_w31(layer.w13_weight.data)
        w13_weight_scale_inv = _swap_w13_to_w31(
            layer.w13_weight_scale_inv.data)
        w2_weight = layer.w2_weight.data
        w2_weight_scale_inv = layer.w2_weight_scale_inv.data
    else:
        w13_weight = layer.w13_weight.data
        w13_weight_scale_inv = layer.w13_weight_scale_inv.data
        w2_weight = layer.w2_weight
        w2_weight_scale_inv = layer.w2_weight_scale_inv

    print(f"[lark] w13_weight.shape: {w13_weight.shape}, w13_weight_scale_inv.shape: {w13_weight_scale_inv.shape}, w2_weight.shape: {w2_weight.shape}, w2_weight_scale_inv.shape: {w2_weight_scale_inv.shape}")

    print(f"[lark] layer.w13_weight dir {dir(layer.w13_weight)}")
    print(f"[lark] layer.w13_weight_scale_inv dir {dir(layer.w13_weight_scale_inv)}")
    print(f"[lark] layer.w2_weight dir {dir(layer.w2_weight)}")
    print(f"[lark] layer.w2_weight_scale_inv dir {dir(layer.w2_weight_scale_inv)}")

    # torch.compile() cannot use Parameter subclasses.
    layer.w13_weight = Parameter(w13_weight, requires_grad=False)
    layer.w13_weight_scale_inv = Parameter(w13_weight_scale_inv,
                                            requires_grad=False)
    layer.w2_weight = Parameter(w2_weight, requires_grad=False)
    layer.w2_weight_scale_inv = Parameter(w2_weight_scale_inv,
                                            requires_grad=False)

    # DeepGemm scales need to be transposed and aligned.  We try to do
    # it ahead of time for performance reasons.
    if self.allow_deep_gemm and not is_blackwell_deep_gemm_used():
        # Lazy import to avoid CUDA initialization problems.
        if _is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = \
                get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous()
        if _is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = \
                get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous()

    if is_blackwell_deep_gemm_used():
        assert layer.weight_block_size is not None
        # Re-quantise the expert weights so their scales are UE8M0.
        block_sz = tuple(layer.weight_block_size)
        requant_weight_ue8m0_inplace(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            block_sz,
        )
        requant_weight_ue8m0_inplace(
            layer.w2_weight.data,
            layer.w2_weight_scale_inv.data,
            block_sz,
        )

        # Ensure column-major TMA alignment expected by DeepGEMM.
        if _is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(
                layer.w13_weight_scale_inv).contiguous()
        if _is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(
                layer.w2_weight_scale_inv).contiguous()