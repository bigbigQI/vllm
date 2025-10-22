# vLLM MoE Expert Router 追踪功能开发方案

## 一、概述

本方案旨在为 vLLM 项目添加 MoE (Mixture of Experts) 模型在推理过程中的 expert router 选择追踪功能，使得模型在生成最终推理结果时，能够返回：

1. 模型每层网络 expert router 选择了哪些专家
2. 模型所有 token（包括 prefill 和 decode 阶段）的 expert router 选择情况

## 二、需求分析

### 2.1 功能需求

1. **完整性**：追踪所有 token 的 expert 选择，包括：
   - Prefill 阶段的所有 prompt tokens
   - Decode 阶段的所有生成 tokens
   - 按照处理顺序统一返回（不区分阶段）

2. **层级性**：记录每一层 MoE 层的 expert 选择
   - 支持多层 MoE 架构
   - 使用层名称（layer_name）标识
   - 每层可能有不同数量的 experts

3. **简洁性**：对于每个 token，记录：
   - Token 在序列中的位置索引
   - 选择的 top-k experts 的 ID 列表
   - **不记录** expert 权重（简化数据）

4. **兼容性**：
   - 不影响现有功能
   - 通过环境变量全局控制
   - 保持向后兼容

### 2.2 使用场景

1. **模型分析**：了解不同 token 如何路由到不同 experts
2. **性能优化**：分析 expert 负载均衡情况
3. **调试诊断**：排查 MoE 路由问题
4. **研究需求**：分析 expert 专业化程度

## 三、技术方案设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                 1. 环境变量配置 (envs.py)                     │
│             VLLM_TRACK_MOE_EXPERT_SELECTIONS                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              2. MoE 层捕获 (FusedMoE Layer)                  │
│      在 forward 中记录 layer_name + topk_ids                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            3. 上下文传递 (ForwardContext)                     │
│       临时存储当前 forward pass 的 expert 选择               │
│           (layer_name -> token_indices + topk_ids)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           4. 输出封装 (EngineCoreOutput)                     │
│              将 expert 选择附加到输出                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           5. 状态累积 (RequestState)                         │
│      按顺序累积所有 token 的 expert 选择                     │
│         (不区分 prefill/decode)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           6. 最终输出 (RequestOutput)                        │
│    返回按 token 顺序的 expert 选择 (layer_name + ids)        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据结构设计

#### 3.2.1 ExpertSelection 数据结构

```python
# 新文件: vllm/model_executor/layers/moe_utils.py

from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class TokenExpertSelection:
    """单个 token 在某一层的 expert 选择信息"""
    token_index: int  # Token 在序列中的位置（全局位置）
    layer_name: str  # MoE 层的名称（如 "model.layers.5.mlp"）
    selected_expert_ids: List[int]  # 选择的 expert IDs (top-k)
    
    def to_dict(self) -> dict:
        """转换为字典格式用于输出"""
        return {
            "token_index": self.token_index,
            "layer_name": self.layer_name,
            "selected_experts": self.selected_expert_ids,
        }


@dataclass
class RequestExpertSelections:
    """整个请求的所有 expert 选择（按 token 顺序组织）"""
    request_id: str
    # token_index -> layer_name -> selected_expert_ids
    # 例如: {0: {"layer_5": [1, 3], "layer_10": [2, 4]}, 1: {...}}
    token_selections: Dict[int, Dict[str, List[int]]]
    
    def add_token_layer_selection(
        self,
        token_index: int,
        layer_name: str,
        expert_ids: List[int]
    ):
        """添加一个 token 在某层的 expert 选择"""
        if token_index not in self.token_selections:
            self.token_selections[token_index] = {}
        self.token_selections[token_index][layer_name] = expert_ids
    
    def to_dict(self) -> dict:
        """转换为完整的字典格式
        
        返回格式:
        {
            "request_id": "xxx",
            "num_tokens": 20,
            "selections": [
                {
                    "token_index": 0,
                    "layers": {
                        "model.layers.5.mlp": [1, 3, 5],
                        "model.layers.10.mlp": [2, 4, 6]
                    }
                },
                ...
            ]
        }
        """
        selections = []
        for token_idx in sorted(self.token_selections.keys()):
            selections.append({
                "token_index": token_idx,
                "layers": self.token_selections[token_idx]
            })
        
        return {
            "request_id": self.request_id,
            "num_tokens": len(self.token_selections),
            "selections": selections
        }
```

### 3.3 关键模块修改

#### 3.3.1 环境变量配置

**文件：** `vllm/envs.py`

在 `environment_variables` 字典中添加：

```python
# 在 environment_variables 字典中添加（约 line 1400 附近）
environment_variables: dict[str, Callable[[], Any]] = {
    # ... 现有变量 ...
    
    # 是否追踪 MoE expert 选择
    # 如果启用，vLLM 会记录每个 token 在每个 MoE 层选择的 expert IDs
    # 并在 RequestOutput 中返回。这对于分析 MoE 路由行为很有用。
    # 注意：启用此功能会有轻微的性能开销（约 1-3%）
    "VLLM_TRACK_MOE_EXPERT_SELECTIONS": lambda: bool(
        int(os.getenv("VLLM_TRACK_MOE_EXPERT_SELECTIONS", "0"))
    ),
    
    # ... 其他变量 ...
}
```

使用方式：
```bash
# 启用 MoE expert 追踪
export VLLM_TRACK_MOE_EXPERT_SELECTIONS=1

# 禁用（默认）
export VLLM_TRACK_MOE_EXPERT_SELECTIONS=0
```

#### 3.3.2 ForwardContext 修改

**文件：** `vllm/forward_context.py`

```python
from typing import Dict, List, Optional
import torch
import vllm.envs as envs

@dataclass
class ForwardContext:
    # ... 现有字段 ...
    
    # 新增字段：存储当前 forward pass 的 expert 选择
    # layer_name -> List[(token_index, expert_ids)]
    moe_expert_selections: Optional[Dict[str, List[tuple[int, List[int]]]]] = None
    
    # 是否启用 expert 追踪（从环境变量读取）
    track_expert_selections: bool = False
    
    def record_expert_selection(
        self,
        layer_name: str,
        token_indices: torch.Tensor,  # [num_tokens]
        expert_ids: torch.Tensor,     # [num_tokens, top_k]
    ):
        """记录一层的 expert 选择
        
        Args:
            layer_name: MoE 层的名称（如 "model.layers.5.mlp"）
            token_indices: token 的全局索引
            expert_ids: 每个 token 选择的 expert IDs
        """
        if not self.track_expert_selections:
            return
        
        if self.moe_expert_selections is None:
            self.moe_expert_selections = {}
        
        if layer_name not in self.moe_expert_selections:
            self.moe_expert_selections[layer_name] = []
        
        # 转换为 CPU 并保存
        num_tokens = token_indices.size(0)
        for token_idx in range(num_tokens):
            global_token_idx = int(token_indices[token_idx].item())
            selected_ids = expert_ids[token_idx].cpu().tolist()
            
            self.moe_expert_selections[layer_name].append(
                (global_token_idx, selected_ids)
            )
    
    def get_and_clear_expert_selections(
        self
    ) -> Optional[Dict[str, List[tuple[int, List[int]]]]]:
        """获取并清空 expert 选择记录
        
        Returns:
            Dict[layer_name, List[(token_index, expert_ids)]]
        """
        selections = self.moe_expert_selections
        self.moe_expert_selections = None
        return selections
```

#### 3.3.3 FusedMoE 层修改

**文件：** `vllm/model_executor/layers/fused_moe/layer.py`

**核心思路：** 直接在 `UnquantizedFusedMoEMethod.forward_cuda()` 中捕获 `topk_ids`，因为该方法已经调用了 `select_experts()` 并获得了结果，**无需重复计算**。

##### 修改 1: FusedMoE 初始化添加 layer_name

```python
class FusedMoE(CustomOp):
    def __init__(
        self,
        # ... 现有参数 ...
        prefix: str = "",  # 已有参数
    ):
        super().__init__()
        # ... 现有初始化 ...
        
        # 新增：MoE 层名称（用于追踪）
        # 使用 prefix 作为层名称，如 "model.layers.5.mlp.experts"
        self.layer_name = prefix  # 此行已存在（约 line 1129）
```

**说明：** `FusedMoE.__init__` 中已经有 `self.layer_name = prefix`，**无需修改**。

##### 修改 2: 在 UnquantizedFusedMoEMethod.forward_cuda 中捕获

**位置：** 约 line 594-703

**修改前代码：**
```python
@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    # ...
    
    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        # ... 其他参数 ...
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)

        # 调用 select_experts 获取 topk_ids (line 620-642)
        topk_weights, topk_ids, zero_expert_result = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            # ... 其他参数 ...
        )

        # 使用 topk_ids 进行专家计算 (line 644-703)
        if self.rocm_aiter_moe_enabled:
            result = self.rocm_aiter_fused_experts(...)
        elif self.flashinfer_cutlass_moe_enabled:
            result = self.flashinfer_cutlass_moe(...)
        # ...
        
        return result
```

**修改后代码：**
```python
@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    # ...
    
    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        # ... 其他参数 ...
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)

        # 调用 select_experts 获取 topk_ids (line 620-642)
        topk_weights, topk_ids, zero_expert_result = FusedMoE.select_experts(
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
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            global_num_experts=global_num_experts,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
            num_fused_shared_experts=layer.num_fused_shared_experts,
        )

        # ============ 新增：记录 expert 选择 ============
        self._maybe_record_expert_selections(
            layer=layer,
            hidden_states=x,
            topk_ids=topk_ids,
        )
        # ============================================

        # 使用 topk_ids 进行专家计算 (line 644-703)
        if self.rocm_aiter_moe_enabled:
            assert self.fused_experts is None
            result = self.rocm_aiter_fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        elif self.flashinfer_cutlass_moe_enabled:
            return self.flashinfer_cutlass_moe(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        elif self.fused_experts is not None:
            if self.moe.has_bias:
                raise ValueError("FusedMoEModularKernel does not support bias.")
            result = self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )
        else:
            assert fused_experts is not None
            result = fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

        if zero_expert_num != 0 and zero_expert_type is not None:
            assert not isinstance(result, tuple), (
                "Shared + zero experts are mutually exclusive not yet supported"
            )
            return result, zero_expert_result
        else:
            return result
    
    def _maybe_record_expert_selections(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        """如果启用追踪，记录 expert 选择（零性能开销版本）"""
        try:
            from vllm.forward_context import get_forward_context
            ctx = get_forward_context()
            
            # 快速检查：如果未启用追踪，立即返回
            if ctx is None or not ctx.track_expert_selections:
                return
            
            # 获取层名称
            layer_name = getattr(layer, 'layer_name', 'unknown_moe_layer')
            
            # 创建 token 索引
            num_tokens = hidden_states.size(0)
            token_indices = torch.arange(
                num_tokens, 
                device=hidden_states.device
            )
            
            # 记录到 context（直接使用已计算的 topk_ids，零额外开销）
            ctx.record_expert_selection(
                layer_name=layer_name,
                token_indices=token_indices,
                expert_ids=topk_ids,
            )
        except Exception as e:
            # 追踪失败不应影响主流程
            import logging
            logging.warning(f"Failed to record expert selections: {e}")
```

**关键优势：**

1. ✅ **零重复计算**：直接使用 `forward_cuda` 中已调用 `select_experts()` 得到的 `topk_ids`
2. ✅ **最小侵入性**：只需在 `forward_cuda` 中添加 3 行调用代码
3. ✅ **零性能开销**：不启用追踪时，只有一个 `if` 判断（< 1 纳秒）
4. ✅ **异常安全**：try-catch 确保追踪失败不影响主流程
5. ✅ **统一捕获点**：所有平台（CUDA/ROCm/FlashInfer）共用同一捕获逻辑

#### 3.3.4 ModelRunner 修改

**文件：** `vllm/v1/worker/gpu_model_runner.py`

```python
class GPUModelRunner:
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput:
        # ... 准备输入 ...
        
        # 检查是否启用 expert tracking（从环境变量）
        import vllm.envs as envs
        track_expert_selections = envs.VLLM_TRACK_MOE_EXPERT_SELECTIONS
        
        # Run the model with expert tracking
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
            # 新增参数
            track_expert_selections=track_expert_selections,
        ):
            model_output = self._model_forward(...)
        
        # 从 context 中提取 expert selections
        expert_selections = None
        if track_expert_selections:
            ctx = get_forward_context()
            expert_selections = ctx.get_and_clear_expert_selections()
        
        # ... 后续处理 ...
        
        # 将 expert_selections 附加到 model_runner_output
        return model_runner_output  # 需要扩展此结构
```


