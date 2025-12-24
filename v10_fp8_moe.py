def process_weights_after_loading(self, layer: Module) -> None:
    assert self.quant_config.activation_scheme == "dynamic"

    if self.flashinfer_moe_backend is not None:
        w13_weight = swap_w13_to_w31(layer.w13_weight.data)
        w13_weight_scale_inv = swap_w13_to_w31(
            layer.w13_weight_scale_inv.data)
        w2_weight = layer.w2_weight.data
        w2_weight_scale_inv = layer.w2_weight_scale_inv.data
    else:
        w13_weight = layer.w13_weight.data
        w13_weight_scale_inv = layer.w13_weight_scale_inv.data
        w2_weight = layer.w2_weight
        w2_weight_scale_inv = layer.w2_weight_scale_inv

    # torch.compile() cannot use Parameter subclasses.
    layer.w13_weight = Parameter(w13_weight, requires_grad=False)
    layer.w13_weight_scale_inv = Parameter(w13_weight_scale_inv,
                                            requires_grad=False)
    layer.w2_weight = Parameter(w2_weight, requires_grad=False)
    layer.w2_weight_scale_inv = Parameter(w2_weight_scale_inv,
                                            requires_grad=False)

    # DeepGemm scales need to be transposed and aligned. We try to do
    # it ahead of time for performance reasons.
    if self.allow_deep_gemm and not is_deep_gemm_e8m0_used():
        if expert_weight_is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv)
        if expert_weight_is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv)

    if is_deep_gemm_e8m0_used():
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
        if expert_weight_is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(
                layer.w13_weight_scale_inv)
        if expert_weight_is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(
                layer.w2_weight_scale_inv)