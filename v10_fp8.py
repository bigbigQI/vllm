def process_weights_after_loading(self, layer: Module) -> None:






























    weight, weight_scale = process_fp8_weight_block_strategy(layer.weight, layer.weight_scale_inv)
    del layer.weight_scale_inv


# Update layer with new values.
layer.weight = Parameter(weight.data, requires_grad=False)
layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
layer.input_scale = Parameter(
    input_scale,
    requires_grad=False) if input_scale is not None else None


maybe_post_process_fp8_weight_block(layer, self.cutlass_block_fp8_supported)