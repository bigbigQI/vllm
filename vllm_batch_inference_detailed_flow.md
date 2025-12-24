# vLLM Batch推理详细流程分析（从入口到执行）

本文档从 `vllm/entrypoints/llm.py` 入口开始，详细追踪当你传入一个batch的prompts（例如4个prompts）时，vLLM是如何处理prefill和decode阶段的。每一步都会结合代码进行说明。

---

## 场景设定

假设我们有这样的代码：

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 准备batch prompts
prompts = [
    "Tell me a story",      # Request 1: 4 tokens
    "What is AI?",          # Request 2: 3 tokens
    "Explain quantum",      # Request 3: 2 tokens
    "Hello world",          # Request 4: 2 tokens
]

# 采样参数
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

# 开始推理
outputs = llm.generate(prompts, sampling_params)
```

---

## 第一阶段：入口 - LLM.generate()

**文件路径**: `vllm/entrypoints/llm.py:364-430`

### 1.1 generate() 方法

```python:vllm/entrypoints/llm.py
def generate(
    self,
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    priority: list[int] | None = None,
) -> list[RequestOutput]:
    """Generates the completions for the input prompts.
    
    This class automatically batches the given prompts, considering
    the memory constraint. For the best performance, put all of your prompts
    into a single list and pass it to this method.
    """
    # 第一步：验证runner类型
    model_config = self.model_config
    runner_type = model_config.runner_type
    if runner_type != "generate":
        raise ValueError("LLM.generate() is only supported for generative models.")
    
    # 第二步：获取默认sampling params（如果没提供）
    if sampling_params is None:
        sampling_params = self.get_default_sampling_params()
    
    # 第三步：验证并添加所有requests到引擎
    self._validate_and_add_requests(
        prompts=prompts,
        params=sampling_params,
        use_tqdm=use_tqdm,
        lora_request=lora_request,
        priority=priority,
    )
    
    # 第四步：运行引擎直到所有requests完成
    outputs = self._run_engine(use_tqdm=use_tqdm)
    return self.engine_class.validate_outputs(outputs, RequestOutput)
```

**这一步做了什么：**
- 收到4个prompts
- 将它们转化为requests并添加到引擎
- 运行引擎的event loop直到所有请求完成

---

## 第二阶段：添加Requests到引擎

### 2.1 _validate_and_add_requests()

**文件路径**: `vllm/entrypoints/llm.py:1515-1564`

```python:vllm/entrypoints/llm.py
def _validate_and_add_requests(
    self,
    prompts: PromptType | Sequence[PromptType] | DataPrompt,
    params: SamplingParams | Sequence[SamplingParams] | ...,
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: Sequence[LoRARequest] | LoRARequest | None,
    priority: list[int] | None = None,
) -> None:
    # 第一步：确保prompts是列表
    if isinstance(prompts, (str, dict)):
        prompts = [prompts]  # 单个prompt转为列表
    
    num_requests = len(prompts)  # 我们的case: 4
    
    # 第二步：验证params和lora_request的长度匹配
    if isinstance(params, Sequence) and len(params) != num_requests:
        raise ValueError("The lengths of prompts and params must be the same.")
    
    # 第三步：循环添加每个request
    it = prompts
    if use_tqdm:
        tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
        it = tqdm_func(it, desc="Adding requests")
    
    for i, prompt in enumerate(it):
        # 验证多模态数据
        if isinstance(prompt, dict):
            self._validate_mm_data_and_uuids(
                prompt.get("multi_modal_data"), 
                prompt.get("multi_modal_uuids")
            )
        
        # 添加单个request
        self._add_request(
            prompt,
            params[i] if isinstance(params, Sequence) else params,
            lora_request=lora_request[i] if isinstance(lora_request, Sequence) else lora_request,
            priority=priority[i] if priority else 0,
        )
```

**这一步做了什么：**
- 对于我们的4个prompts，循环调用 `_add_request()` 4次
- 每个prompt变成一个独立的request

---

### 2.2 _add_request()

**文件路径**: `vllm/entrypoints/llm.py:1640-1666`

```python:vllm/entrypoints/llm.py
def _add_request(
    self,
    prompt: PromptType,
    params: SamplingParams | PoolingParams,
    lora_request: LoRARequest | None = None,
    priority: int = 0,
) -> None:
    # 第一步：提取prompt文本
    prompt_text, _, _ = get_prompt_components(prompt)
    
    # 第二步：生成unique request_id
    request_id = str(next(self.request_counter))  # "0", "1", "2", "3"
    
    # 第三步：处理输入（tokenization等）
    engine_request, tokenization_kwargs = self._process_inputs(
        request_id,
        prompt,
        params,
        lora_request=lora_request,
        priority=priority,
    )
    
    # 第四步：将request添加到LLMEngine
    self.llm_engine.add_request(
        request_id,
        engine_request,  # EngineCoreRequest对象
        params,
        lora_request=lora_request,
        tokenization_kwargs=tokenization_kwargs,
        priority=priority,
        prompt_text=prompt_text,
    )
```

**这一步做了什么：**
- 为每个prompt生成唯一的request_id: "0", "1", "2", "3"
- 调用 `_process_inputs()` 进行tokenization
- 将tokenized request添加到LLMEngine

---

### 2.3 _process_inputs() - Tokenization

**文件路径**: `vllm/entrypoints/llm.py:1613-1638`

```python:vllm/entrypoints/llm.py
def _process_inputs(
    self,
    request_id: str,
    engine_prompt: PromptType,
    params: SamplingParams | PoolingParams,
    *,
    lora_request: LoRARequest | None,
    priority: int,
) -> tuple[EngineCoreRequest, dict[str, Any]]:
    """Use the Processor to process inputs for LLMEngine."""
    tokenization_kwargs: dict[str, Any] = {}
    _validate_truncation_size(
        self.model_config.max_model_len,
        params.truncate_prompt_tokens,
        tokenization_kwargs,
    )
    
    # 调用Processor进行tokenization
    engine_request = self.processor.process_inputs(
        request_id,
        engine_prompt,
        params,
        lora_request=lora_request,
        tokenization_kwargs=tokenization_kwargs,
        priority=priority,
    )
    return engine_request, tokenization_kwargs
```

**Processor.process_inputs() 会做：**
- 使用tokenizer将文本转换为token IDs
  - "Tell me a story" → [1, 24948, 592, 263, 5828]
  - "What is AI?" → [1, 1724, 338, 319, 29902, 29973]
  - "Explain quantum" → [1, 12027, 7420, 12101]
  - "Hello world" → [1, 15043, 3186]
- 创建 `EngineCoreRequest` 对象

**EngineCoreRequest 结构**:

```python:vllm/v1/engine/__init__.py
class EngineCoreRequest(msgspec.Struct):
    request_id: str                    # "0", "1", "2", "3"
    prompt_token_ids: list[int]        # tokenized prompt
    mm_features: list[...]             # 多模态特征（如果有）
    sampling_params: SamplingParams    # 采样参数
    pooling_params: PoolingParams | None
    eos_token_id: int | None          # EOS token ID
    arrival_time: float               # 到达时间戳
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None
    priority: int = 0
```

---

### 2.4 LLMEngine.add_request()

**文件路径**: `vllm/v1/engine/llm_engine.py:227-286`

```python:vllm/v1/engine/llm_engine.py
def add_request(
    self,
    request_id: str,
    prompt: EngineCoreRequest | PromptType,
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    prompt_text: str | None = None,
) -> None:
    # 验证request_id类型
    if not isinstance(request_id, str):
        raise TypeError(f"request_id must be a string, got {type(request_id)}")
    
    # prompt应该已经是EngineCoreRequest了
    if isinstance(prompt, EngineCoreRequest):
        request = prompt
    else:
        # fallback: 如果不是，再次处理
        request = self.processor.process_inputs(...)
        prompt_text = prompt if isinstance(prompt, str) else prompt.get("prompt")
    
    n = params.n if isinstance(params, SamplingParams) else 1
    
    if n == 1:
        # 正常情况：n=1（不需要多个候选）
        
        # 第一步：将request添加到OutputProcessor
        # OutputProcessor负责detokenization和输出处理
        self.output_processor.add_request(request, prompt_text, None, 0)
        
        # 第二步：将request添加到EngineCore
        # EngineCore负责调度和模型执行
        self.engine_core.add_request(request)
        return
    
    # 如果n>1，需要fan out多个child requests
    # （我们的case不涉及这个）
```

**这一步做了什么：**
- 将每个request分别添加到：
  1. **OutputProcessor**: 负责输出处理、detokenization
  2. **EngineCore**: 负责调度、batch管理、模型执行

现在，4个requests都在EngineCore的队列中等待调度。

---

## 第三阶段：运行引擎 - _run_engine()

**文件路径**: `vllm/entrypoints/llm.py:1668-1717`

```python:vllm/entrypoints/llm.py
def _run_engine(
    self, *, use_tqdm: bool | Callable[..., tqdm] = True
) -> list[RequestOutput | PoolingRequestOutput]:
    # 初始化进度条
    if use_tqdm:
        num_requests = self.llm_engine.get_num_unfinished_requests()  # 4
        tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
        pbar = tqdm_func(
            total=num_requests,
            desc="Processed prompts",
            dynamic_ncols=True,
        )
    
    # 运行引擎的主循环
    outputs: list[RequestOutput | PoolingRequestOutput] = []
    total_in_toks = 0
    total_out_toks = 0
    
    # ========== 核心循环 ==========
    while self.llm_engine.has_unfinished_requests():
        # 每次iteration调用一次step()
        step_outputs = self.llm_engine.step()
        
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                if use_tqdm:
                    # 更新进度条
                    pbar.update(1)
    
    if use_tqdm:
        pbar.close()
    
    return outputs
```

**这一步做了什么：**
- 进入主循环，不断调用 `llm_engine.step()`
- 每次step会：
  1. 调度一些tokens（可能是prefill或decode）
  2. 执行模型forward
  3. 采样生成新tokens
  4. 返回已完成的requests的输出
- 循环直到所有4个requests都完成

---

## 第四阶段：引擎Step - LLMEngine.step()

**文件路径**: `vllm/v1/engine/llm_engine.py:288-319`

```python:vllm/v1/engine/llm_engine.py
def step(self) -> list[RequestOutput] | list[PoolingRequestOutput]:
    # 如果需要执行dummy batch（DP相关），先处理
    if self.should_execute_dummy_batch:
        self.should_execute_dummy_batch = False
        self.engine_core.execute_dummy_batch()
        return []
    
    # ========== 第1步：从EngineCore获取输出 ==========
    outputs = self.engine_core.get_output()
    # outputs是EngineCoreOutputs，包含：
    #   - outputs: list[EngineCoreOutput]  # 每个request的输出
    #   - scheduler_stats: SchedulerStats   # 调度统计
    #   - timestamp: float                 # 时间戳
    
    # ========== 第2步：处理EngineCoreOutputs ==========
    iteration_stats = IterationStats() if self.log_stats else None
    processed_outputs = self.output_processor.process_outputs(
        outputs.outputs,
        engine_core_timestamp=outputs.timestamp,
        iteration_stats=iteration_stats,
    )
    # processed_outputs包含：
    #   - request_outputs: list[RequestOutput]  # 用户可见的输出
    #   - reqs_to_abort: list[str]            # 需要abort的requests
    
    # ========== 第3步：Abort已完成的requests ==========
    self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
    
    # ========== 第4步：记录统计信息 ==========
    if self.logger_manager is not None:
        self.logger_manager.record(
            scheduler_stats=outputs.scheduler_stats,
            iteration_stats=iteration_stats,
            mm_cache_stats=self.processor.stat_mm_cache(),
        )
        self.do_log_stats_with_interval()
    
    return processed_outputs.request_outputs
```

**这一步做了什么：**
- 从EngineCore获取本次iteration的输出
- 通过OutputProcessor进行detokenization
- 返回完成的RequestOutput对象

**重点**：`engine_core.get_output()` 内部会触发调度和模型执行！

---

## 第五阶段：EngineCore的核心循环

**文件路径**: `vllm/v1/engine/core.py:309-330`

### 5.1 EngineCore.step()

```python:vllm/v1/engine/core.py
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output.
    
    Returns tuple of outputs and a flag indicating whether the model
    was executed.
    """
    
    # ========== 第1步：检查是否有requests ==========
    if not self.scheduler.has_requests():
        return {}, False
    
    # ========== 第2步：调度 ==========
    scheduler_output = self.scheduler.schedule()
    # scheduler_output包含：
    #   - scheduled_new_reqs: list[Request]        # 新调度的requests
    #   - scheduled_running_reqs: list[Request]    # 继续运行的requests
    #   - num_scheduled_tokens: dict[str, int]    # 每个request分配的token数
    #   - total_num_scheduled_tokens: int         # 总token数
    #   - ...
    
    # ========== 第3步：执行模型 ==========
    with self.log_error_detail(scheduler_output):
        model_output = self.model_executor.execute_model(scheduler_output)
    # model_output是ModelRunnerOutput，包含：
    #   - sampled_token_ids: 采样的新tokens
    #   - logprobs: log概率
    #   - prompt_logprobs_dict: prompt的logprobs
    #   - ...
    
    assert isinstance(model_output, ModelRunnerOutput)
    
    # ========== 第4步：更新scheduler状态 ==========
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

**这一步做了什么：**
1. **调度器决定**：哪些requests的哪些tokens在这次iteration执行
2. **模型执行**：forward pass计算
3. **更新状态**：更新每个request的进度

---

## 第六阶段：Scheduler调度详解

**文件路径**: `vllm/v1/core/sched/scheduler.py:176-510`

### 6.1 Scheduler.schedule()

```python:vllm/v1/core/sched/scheduler.py
def schedule(self) -> SchedulerOutput:
    # ========== vLLM V1的调度哲学 ==========
    # NOTE(woosuk) on the scheduling algorithm:
    # There's no "decoding phase" nor "prefill phase" in the scheduler.
    # Each request just has the num_computed_tokens and
    # num_tokens_with_spec. num_tokens_with_spec =
    # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
    # At each step, the scheduler tries to assign tokens to the requests
    # so that each request's num_computed_tokens can catch up its
    # num_tokens_with_spec.
    
    scheduled_new_reqs: list[Request] = []        # 新开始的requests
    scheduled_resumed_reqs: list[Request] = []    # 恢复的requests
    scheduled_running_reqs: list[Request] = []    # 继续运行的requests
    preempted_reqs: list[Request] = []            # 被抢占的requests
    
    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens  # 例如: 2048
    
    # ========== 第1步：调度RUNNING requests ==========
    req_index = 0
    while req_index < len(self.running):
        request = self.running[req_index]
        
        # 计算这个request需要多少新tokens
        num_new_tokens = request.num_tokens - request.num_computed_tokens
        
        # 检查token budget是否够用
        if num_new_tokens > token_budget:
            # 如果超出budget，进行chunked prefill
            num_new_tokens = token_budget
        
        # 检查KV cache空间是否足够
        num_new_blocks = self.kv_cache_manager.get_num_required_blocks(
            request, num_new_tokens
        )
        new_blocks = self.kv_cache_manager.allocate_blocks(...)
        
        if new_blocks is None:
            # KV cache不够，需要preempt一些requests
            break
        
        # 调度这个request
        scheduled_running_reqs.append(request)
        req_to_new_blocks[request.request_id] = new_blocks
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1
    
    # ========== 第2步：调度WAITING requests ==========
    # （如果token budget还有剩余）
    while len(self.waiting) > 0 and token_budget > 0:
        request = self.waiting[0]
        
        # 计算需要多少tokens（可能是整个prompt或chunk）
        num_prompt_tokens = len(request.prompt_token_ids)
        num_new_tokens = min(num_prompt_tokens, token_budget)
        
        # 分配KV cache blocks
        new_blocks = self.kv_cache_manager.allocate_blocks(...)
        if new_blocks is None:
            break  # 没有足够的KV cache
        
        # 调度这个新request
        scheduled_new_reqs.append(request)
        req_to_new_blocks[request.request_id] = new_blocks
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        
        # 从waiting移到running
        self.waiting.popleft()
        self.running.append(request)
    
    # ========== 第3步：构造SchedulerOutput ==========
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_resumed_reqs=scheduled_resumed_reqs,
        scheduled_running_reqs=scheduled_running_reqs,
        preempted_reqs=preempted_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        scheduled_new_reqs_data=...,
        scheduled_resumed_reqs_data=...,
        scheduled_running_reqs_data=...,
        ...
    )
```

**调度逻辑示例**：

假设 `max_num_batched_tokens=2048`：

**Iteration 1（所有requests刚到达）：**
```
WAITING队列: [Req0(4 tokens), Req1(3 tokens), Req2(2 tokens), Req3(2 tokens)]
Token Budget: 2048

调度结果：
  Req0: 4 tokens (prefill, position 0-3)
  Req1: 3 tokens (prefill, position 0-2)
  Req2: 2 tokens (prefill, position 0-1)
  Req3: 2 tokens (prefill, position 0-1)
  
Total: 11 tokens (全部是prefill)
Status: 所有requests移到RUNNING队列，num_computed_tokens = [4, 3, 2, 2]
```

**Iteration 2（所有requests在decode）：**
```
RUNNING队列: [Req0(4→5), Req1(3→4), Req2(2→3), Req3(2→3)]
Token Budget: 2048

调度结果：
  Req0: 1 token (decode, position 4)
  Req1: 1 token (decode, position 3)
  Req2: 1 token (decode, position 2)
  Req3: 1 token (decode, position 2)
  
Total: 4 tokens (全部是decode)
Status: num_computed_tokens = [5, 4, 3, 3]
```

**Iteration 3（继续decode）：**
```
类似Iteration 2，每个request生成1个新token
```

---

## 第七阶段：模型执行 - GPUModelRunner

**文件路径**: `vllm/v1/worker/gpu_model_runner.py:2408-2692`

### 7.1 GPUModelRunner.execute_model()

```python:vllm/v1/worker/gpu_model_runner.py
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
    
    # ========== 第1步：Preprocess（准备输入） ==========
    with record_function_or_nullcontext("Preprocess"):
        with self.synchronize_input_prep():
            # 更新persistent batch状态
            self._update_states(scheduler_output)
            
            if not scheduler_output.total_num_scheduled_tokens:
                # 没有tokens需要处理
                return EMPTY_MODEL_RUNNER_OUTPUT
            
            # ========== 准备decoder inputs ==========
            (
                attn_metadata,        # attention metadata（prefill/decode分离）
                logits_indices,       # 需要计算logits的位置
                spec_decode_metadata,
                num_scheduled_tokens_np,
                spec_decode_common_attn_metadata,
                max_query_len,
                ubatch_slices,
                num_tokens_across_dp,
                use_cascade_attn,
            ) = self._prepare_inputs(scheduler_output)
        
        # ========== 计算实际的input token数 ==========
        # （可能包含DP padding）
        dp_rank = self.parallel_config.data_parallel_rank
        if num_tokens_across_dp is not None:
            num_input_tokens = int(num_tokens_across_dp[dp_rank].item())
        else:
            num_input_tokens = self._get_num_input_tokens(
                scheduler_output.total_num_scheduled_tokens
            )
        
        # ========== Preprocess：准备input_ids, positions等 ==========
        (
            num_scheduled_tokens,  # 实际调度的token数
            input_ids,            # [num_tokens] tensor
            inputs_embeds,        # 如果使用embeddings
            positions,            # [num_tokens] position tensor
            intermediate_tensors,
            model_kwargs,
        ) = self._preprocess(
            scheduler_output, num_input_tokens, intermediate_tensors
        )
        
        # ========== 确定batch descriptor（用于CUDA graph） ==========
        uniform_decode = (max_query_len == self.uniform_decode_query_len) and \
                        (num_scheduled_tokens == self.input_batch.num_reqs * max_query_len)
        batch_descriptor = BatchDescriptor(
            num_tokens=num_input_tokens,
            uniform_decode=uniform_decode,
            has_lora=len(self.input_batch.lora_id_to_lora_request) > 0,
        )
        cudagraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(batch_descriptor, use_cascade_attn)
    
    # ========== 第2步：Forward（模型前向传播） ==========
    with (
        set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
        ),
        record_function_or_nullcontext("Forward"),
    ):
        # ========== 调用模型的forward ==========
        model_output = self._model_forward(
            input_ids=input_ids,      # [num_tokens]
            positions=positions,       # [num_tokens]
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
    
    # ========== 第3步：Postprocess（处理输出） ==========
    with record_function_or_nullcontext("Postprocess"):
        # 如果不是最后一层PP rank，返回intermediate tensors
        if not get_pp_group().is_last_rank:
            assert isinstance(hidden_states, IntermediateTensors)
            return hidden_states
        
        # ========== 提取logits ==========
        hidden_states = model_output
        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        # logits shape: [num_reqs, vocab_size]
    
    # ========== 第4步：Sample（采样生成tokens） ==========
    with record_function_or_nullcontext("Sample"):
        sampler_output = self._sample(logits, spec_decode_metadata)
        # sampler_output包含：
        #   - sampled_token_ids: [num_reqs, 1]  # 采样的token IDs
        #   - logprobs: log概率（如果需要）
    
    # ========== 第5步：Bookkeep（记账同步） ==========
    with record_function_or_nullcontext("Bookkeep"):
        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            num_scheduled_tokens,
        )
    
    # ========== 第6步：构造输出 ==========
    output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,  # 新生成的tokens
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
    )
    
    return output
```

---

### 7.2 _prepare_inputs() - 准备输入数据

**这是最关键的函数之一**，负责将scheduler_output转换为模型可以接受的tensor格式。

```python:vllm/v1/worker/gpu_model_runner.py
def _prepare_inputs(
    self, scheduler_output: "SchedulerOutput"
) -> tuple[...]:
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    num_reqs = self.input_batch.num_reqs
    
    # ========== 第1步：获取每个request的token数 ==========
    req_ids = self.input_batch.req_ids  # ["0", "1", "2", "3"]
    tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
    num_scheduled_tokens = np.array(tokens, dtype=np.int32)
    # 例如：[4, 3, 2, 2] 在prefill iteration
    # 或：  [1, 1, 1, 1] 在decode iteration
    
    max_num_scheduled_tokens = max(tokens)
    
    # ========== 第2步：计算request indices ==========
    # E.g., [4, 3, 2, 2] -> [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
    
    # ========== 第3步：计算cumulative token counts和arange ==========
    # cu_num_tokens: [4, 3, 2, 2] -> [4, 7, 9, 11]
    # arange: [0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 1]
    cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
    
    # ========== 第4步：计算positions ==========
    positions_np = self.positions.np[:total_num_scheduled_tokens]
    np.add(
        self.input_batch.num_computed_tokens_cpu[req_indices],
        arange,
        out=positions_np,
    )
    # 例如，如果num_computed_tokens=[0, 0, 0, 0]（第一次iteration）
    # positions = [0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 1]
    
    # ========== 第5步：获取token IDs ==========
    # 从token_ids_cpu中按positions提取
    token_indices = (
        positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    )
    torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        token_indices_tensor,
        out=self.input_ids.cpu[:total_num_scheduled_tokens],
    )
    # input_ids现在包含所有要处理的token IDs
    
    # ========== 第6步：准备attention metadata ==========
    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1:num_reqs+1] = cu_num_tokens
    # query_start_loc: [0, 4, 7, 9, 11]
    # 表示每个request在flattened tensor中的起始位置
    
    self.seq_lens.np[:num_reqs] = (
        self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
    )
    # seq_lens: 每个request的总序列长度
    # 第一次iteration: [4, 3, 2, 2]
    # 第二次iteration: [5, 4, 3, 3]
    
    # ========== 第7步：构造attention metadata ==========
    attn_metadata = self.attn_backend.make_metadata(
        num_actual_tokens=total_num_scheduled_tokens,
        max_query_len=max_num_scheduled_tokens,
        query_start_loc=self.query_start_loc.gpu[:num_reqs+1],
        max_seq_len=int(self.seq_lens.np[:num_reqs].max()),
        seq_lens=self.seq_lens.gpu[:num_reqs],
        block_table=self.input_batch.block_table.gpu_tensor[:num_reqs],
        slot_mapping=self.input_batch.block_table.slot_mapping_gpu[:total_num_scheduled_tokens],
        num_prefill_tokens=...,  # 会计算有多少prefill tokens
        num_decode_tokens=...,   # 会计算有多少decode tokens
        num_prefills=...,        # 有多少个requests在prefill
        num_decodes=...,         # 有多少个requests在decode
    )
    
    return (
        attn_metadata,
        logits_indices,  # 需要计算logits的token位置
        ...,
    )
```

**数据结构示例**：

**Iteration 1（Prefill）：**
```
num_scheduled_tokens = [4, 3, 2, 2]
total = 11 tokens

input_ids (flattened):
  [tok0_r0, tok1_r0, tok2_r0, tok3_r0,  # Request 0
   tok0_r1, tok1_r1, tok2_r1,            # Request 1
   tok0_r2, tok1_r2,                     # Request 2
   tok0_r3, tok1_r3]                     # Request 3
  Shape: [11]

positions:
  [0, 1, 2, 3,  # Request 0
   0, 1, 2,     # Request 1
   0, 1,        # Request 2
   0, 1]        # Request 3
  Shape: [11]

query_start_loc:
  [0, 4, 7, 9, 11]
  
seq_lens:
  [4, 3, 2, 2]

attn_metadata:
  num_prefill_tokens = 11
  num_decode_tokens = 0
  num_prefills = 4
  num_decodes = 0
```

**Iteration 2（Decode）：**
```
num_scheduled_tokens = [1, 1, 1, 1]
total = 4 tokens

input_ids (flattened):
  [sampled_tok_r0,   # Request 0 newly sampled
   sampled_tok_r1,   # Request 1 newly sampled
   sampled_tok_r2,   # Request 2 newly sampled
   sampled_tok_r3]   # Request 3 newly sampled
  Shape: [4]

positions:
  [4,  # Request 0, position 4
   3,  # Request 1, position 3
   2,  # Request 2, position 2
   2]  # Request 3, position 2
  Shape: [4]

query_start_loc:
  [0, 1, 2, 3, 4]
  
seq_lens:
  [5, 4, 3, 3]  # 累积长度

attn_metadata:
  num_prefill_tokens = 0
  num_decode_tokens = 4
  num_prefills = 0
  num_decodes = 4
```

---

### 7.3 模型Forward - Attention层的处理

**文件路径**: `vllm/v1/attention/backends/xformers.py` 等

```python
# 伪代码，展示attention如何分别处理prefill和decode
def attention_forward(
    query: torch.Tensor,           # [num_tokens, num_heads, head_dim]
    key_cache: torch.Tensor,       # KV cache
    value_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
):
    # ========== 分离prefill和decode ==========
    prefill_meta = attn_metadata.prefill_metadata
    decode_meta = attn_metadata.decode_metadata
    
    outputs = []
    
    # ========== 处理Decode tokens ==========
    if decode_meta is not None and decode_meta.num_decode_tokens > 0:
        # Decode: 每个request只有1个query token，但要attend到整个history
        decode_query = query[:decode_meta.num_decode_tokens]
        
        # 使用paged attention进行decode
        decode_output = paged_attention(
            decode_query,              # [num_decodes, num_heads, head_dim]
            key_cache,                 # 从KV cache读取
            value_cache,
            block_table=decode_meta.block_table,
            seq_lens=decode_meta.seq_lens,
        )
        outputs.append(decode_output)
    
    # ========== 处理Prefill tokens ==========
    if prefill_meta is not None and prefill_meta.num_prefill_tokens > 0:
        # Prefill: 多个query tokens做causal attention
        prefill_query = query[decode_meta.num_decode_tokens:]
        
        # 使用varlen attention（FlashAttention）进行prefill
        prefill_output = flash_attn_varlen_func(
            q=prefill_query,
            k=key_cache,  # 从cached context读取（如果有）
            v=value_cache,
            cu_seqlens_q=prefill_meta.query_start_loc,  # [0, 4, 7, 9, 11]
            cu_seqlens_k=prefill_meta.query_start_loc,
            max_seqlen_q=prefill_meta.max_query_len,
            max_seqlen_k=prefill_meta.max_seq_len,
            causal=True,  # Causal masking
        )
        outputs.append(prefill_output)
    
    # ========== 合并输出 ==========
    # 注意：decode在前，prefill在后
    return torch.cat(outputs, dim=0)
```

**Attention的关键点**：

1. **Decode Attention**（单query vs 整个history）：
   - Query shape: `[num_decodes, num_heads, head_dim]`
   - 每个request只有1个query
   - 需要attend到该request的所有历史tokens（从KV cache读取）
   - 使用Paged Attention算法，内存效率高

2. **Prefill Attention**（多query之间causal attention）：
   - Query shape: `[num_prefill_tokens, num_heads, head_dim]`
   - 使用FlashAttention的varlen版本
   - Causal masking：每个token只能attend到之前的tokens
   - 计算密集（compute-bound）

---

## 第八阶段：采样和输出

### 8.1 Sampling

```python:vllm/v1/worker/gpu_model_runner.py
def _sample(
    self,
    logits: torch.Tensor,
    spec_decode_metadata: SpecDecodeMetadata | None,
) -> SamplerOutput:
    # logits shape: [num_reqs, vocab_size]
    
    # 应用temperature, top-p, top-k等采样策略
    sampled_token_ids = self.sampler(
        logits,
        self.input_batch.sampling_metadata,
    )
    # sampled_token_ids shape: [num_reqs, 1]
    
    return SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        logprobs=...,
    )
```

### 8.2 更新Scheduler状态

```python:vllm/v1/core/sched/scheduler.py
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
    # ========== 第1步：更新每个request的状态 ==========
    for req_id, new_token_id in zip(
        model_output.req_ids,
        model_output.sampled_token_ids
    ):
        request = self.requests[req_id]
        
        # 添加新生成的token
        request.output_token_ids.append(new_token_id)
        
        # 更新num_computed_tokens
        request.num_computed_tokens += scheduler_output.num_scheduled_tokens[req_id]
        
        # 检查是否finished
        if self._check_stop(request):
            request.status = RequestStatus.FINISHED_STOPPED
            self.finished_req_ids.add(req_id)
    
    # ========== 第2步：构造EngineCoreOutput ==========
    engine_core_outputs = []
    for req_id in model_output.req_ids:
        request = self.requests[req_id]
        
        output = EngineCoreOutput(
            request_id=req_id,
            new_token_ids=[model_output.sampled_token_ids[req_id]],
            new_logprobs=model_output.logprobs[req_id] if model_output.logprobs else None,
            finish_reason=FinishReason.STOP if request.status == RequestStatus.FINISHED_STOPPED else None,
        )
        engine_core_outputs.append(output)
    
    # ========== 第3步：清理finished requests ==========
    self._remove_finished_requests()
    
    return {
        0: EngineCoreOutputs(
            outputs=engine_core_outputs,
            scheduler_stats=self.make_stats(),
        )
    }
```

---

## 完整流程总结

### 时间轴视角

**T0: 初始化**
```
用户调用: llm.generate(prompts=[4个prompts], sampling_params)
```

**T1: 添加Requests（几乎瞬时）**
```
LLM.generate()
  → _validate_and_add_requests()
    → 循环4次：
      → _add_request()
        → tokenize: "Tell me a story" → [1, 24948, 592, 263, 5828]
        → LLMEngine.add_request()
          → OutputProcessor.add_request()  # 注册detokenizer
          → EngineCore.add_request()       # 加入scheduler队列
```

**T2: Iteration 1 - Prefill所有Requests**
```
LLM._run_engine()
  → while has_unfinished_requests():
    → LLMEngine.step()
      → EngineCore.get_output()
        → EngineCore.step()
          
          → Scheduler.schedule()
            输入: 4个requests，每个都是num_computed_tokens=0
            输出: 
              scheduled_new_reqs = [Req0, Req1, Req2, Req3]
              num_scheduled_tokens = {"0": 4, "1": 3, "2": 2, "3": 2}
              total = 11 tokens
          
          → GPUModelRunner.execute_model(scheduler_output)
            
            → _prepare_inputs()
              构造:
                input_ids: [11个tokens, flattened]
                positions: [0,1,2,3, 0,1,2, 0,1, 0,1]
                attn_metadata:
                  num_prefill_tokens=11
                  num_decode_tokens=0
            
            → _model_forward()
              → model.forward(input_ids, positions, attn_metadata)
                → 每一层:
                  → Attention:
                    - 所有tokens做prefill attention（causal）
                    - 使用FlashAttention varlen
                  → MLP
                  → LayerNorm
                → 输出: hidden_states [11, hidden_dim]
              
              → compute_logits(hidden_states[logits_indices])
                logits_indices = [3, 6, 8, 10]  # 每个request的最后一个token
                logits: [4, vocab_size]
            
            → _sample(logits)
              采样生成新tokens:
                Req0 → token_id: 12345
                Req1 → token_id: 67890
                Req2 → token_id: 11111
                Req3 → token_id: 22222
              返回: sampled_token_ids [4, 1]
            
            返回: ModelRunnerOutput(
              sampled_token_ids=[12345, 67890, 11111, 22222],
              ...
            )
          
          → Scheduler.update_from_output()
            更新状态:
              Req0: num_computed_tokens = 4, output_tokens = [12345]
              Req1: num_computed_tokens = 3, output_tokens = [67890]
              Req2: num_computed_tokens = 2, output_tokens = [11111]
              Req3: num_computed_tokens = 2, output_tokens = [22222]
            
            返回: EngineCoreOutputs(
              outputs=[
                EngineCoreOutput(request_id="0", new_token_ids=[12345]),
                EngineCoreOutput(request_id="1", new_token_ids=[67890]),
                EngineCoreOutput(request_id="2", new_token_ids=[11111]),
                EngineCoreOutput(request_id="3", new_token_ids=[22222]),
              ]
            )
      
      → OutputProcessor.process_outputs()
        Detokenization:
          12345 → " about"
          67890 → " Artificial"
          ...
        
        返回: [
          RequestOutput(request_id="0", outputs=[...], finished=False),
          ...
        ]
```

**T3: Iteration 2 - Decode所有Requests**
```
LLM._run_engine() 继续循环
  → LLMEngine.step()
    → EngineCore.step()
      
      → Scheduler.schedule()
        输入: 4个requests，num_computed_tokens=[4, 3, 2, 2]
        输出:
          scheduled_running_reqs = [Req0, Req1, Req2, Req3]
          num_scheduled_tokens = {"0": 1, "1": 1, "2": 1, "3": 1}
          total = 4 tokens
      
      → GPUModelRunner.execute_model()
        
        → _prepare_inputs()
          构造:
            input_ids: [12345, 67890, 11111, 22222]  # 上次采样的tokens
            positions: [4, 3, 2, 2]
            attn_metadata:
              num_prefill_tokens=0
              num_decode_tokens=4
        
        → _model_forward()
          → model.forward(...)
            → Attention:
              - 使用decode attention（paged attention）
              - 每个token attend到各自的full history
              - 从KV cache读取历史
            → 输出: hidden_states [4, hidden_dim]
          
          → compute_logits(hidden_states)
            logits: [4, vocab_size]
        
        → _sample(logits)
          采样新tokens:
            Req0 → 54321
            Req1 → 98765
            Req2 → 33333
            Req3 → 44444
        
        返回: ModelRunnerOutput(sampled_token_ids=[54321, 98765, 33333, 44444])
      
      → Scheduler.update_from_output()
        更新:
          Req0: num_computed_tokens=5, output_tokens=[12345, 54321]
          Req1: num_computed_tokens=4, output_tokens=[67890, 98765]
          Req2: num_computed_tokens=3, output_tokens=[11111, 33333]
          Req3: num_computed_tokens=3, output_tokens=[22222, 44444]
```

**T4 ~ TN: 继续Decode直到所有Requests完成**
```
每次iteration类似T3
直到：
  - 所有requests生成了max_tokens个tokens，或
  - 遇到了EOS token，或
  - 遇到了stop string

完成后，每个RequestOutput的finished=True
从_run_engine()返回，得到最终的outputs列表
```

---

## Prefill vs Decode的核心区别

### Prefill阶段（Iteration 1）

**特征**：
- 处理完整prompt或prompt chunk
- 每个request可能有不同数量的tokens
- 需要做causal self-attention（每个token只能看到之前的tokens）

**数据流**：
```
Request 0: [tok0, tok1, tok2, tok3] → 4 tokens
Request 1: [tok0, tok1, tok2]       → 3 tokens
Request 2: [tok0, tok1]             → 2 tokens
Request 3: [tok0, tok1]             → 2 tokens

Batch tensor:
  input_ids: [tok0_r0, tok1_r0, tok2_r0, tok3_r0, tok0_r1, tok1_r1, tok2_r1, tok0_r2, tok1_r2, tok0_r3, tok1_r3]
  Shape: [11]

Attention:
  - 使用FlashAttention varlen
  - tok0_r0 attends to: [tok0_r0]
  - tok1_r0 attends to: [tok0_r0, tok1_r0]
  - tok2_r0 attends to: [tok0_r0, tok1_r0, tok2_r0]
  - tok3_r0 attends to: [tok0_r0, tok1_r0, tok2_r0, tok3_r0]
  - (similar for other requests)

KV Cache写入:
  - 所有11个tokens的K和V都写入KV cache

Logits计算:
  - 只计算每个request的最后一个token的logits
  - logits_indices = [3, 6, 8, 10]
  - logits shape: [4, vocab_size]

采样:
  - 4个requests各生成1个新token
```

**性能特点**：
- **Compute-bound**: 大量矩阵乘法（attention计算）
- GPU利用率高
- Throughput高（处理多个tokens）

---

### Decode阶段（Iteration 2+）

**特征**：
- 每个request只处理1个新token（刚采样出来的）
- 需要attend到该request的所有历史tokens
- 从KV cache读取历史

**数据流**：
```
Request 0: [new_tok] → position 4, attend to positions [0,1,2,3,4]
Request 1: [new_tok] → position 3, attend to positions [0,1,2,3]
Request 2: [new_tok] → position 2, attend to positions [0,1,2]
Request 3: [new_tok] → position 2, attend to positions [0,1,2]

Batch tensor:
  input_ids: [new_tok_r0, new_tok_r1, new_tok_r2, new_tok_r3]
  Shape: [4]

Attention:
  - 使用Paged Attention
  - new_tok_r0 reads KV cache for all previous tokens of Req0
  - (similar for other requests)

KV Cache:
  - 读取: 所有历史tokens的K和V
  - 写入: 只有4个新tokens的K和V

Logits计算:
  - 计算所有4个tokens的logits
  - logits shape: [4, vocab_size]

采样:
  - 4个requests各生成1个新token
```

**性能特点**：
- **Memory-bound**: 主要开销是读KV cache
- GPU利用率较低（相比prefill）
- Latency敏感（用户等待每个token）

---

## 混合Batch（Prefill + Decode）

vLLM V1的强大之处在于可以在同一个batch中混合prefill和decode：

**示例Iteration**：
```
Scheduler调度:
  Request 0: prefill 100 tokens (long prompt, chunked)
  Request 1: decode 1 token (position 50)
  Request 2: decode 1 token (position 20)
  Request 3: prefill 50 tokens (new request)

Total: 152 tokens in batch

Data layout:
  input_ids: [decode_tok_r1, decode_tok_r2,  # decode在前
              prefill_toks_r0 (100个),        # prefill在后
              prefill_toks_r3 (50个)]
  Shape: [152]

Attention:
  - decode_tok_r1 和 decode_tok_r2: paged attention
  - prefill_toks_r0: flash attention（causal）
  - prefill_toks_r3: flash attention（causal）

优点:
  - 更好的GPU利用率（混合compute和memory操作）
  - 更灵活的调度
  - 减少decode的等待时间
```

---

## KV Cache管理

### Paged Attention

vLLM使用Paged Attention来管理KV cache：

**概念**：
- KV cache被分成固定大小的blocks（例如16 tokens/block）
- 每个request动态分配需要的blocks
- Blocks可以在requests之间共享（prefix caching）

**Block Table示例**：
```
Request 0 (sequence length 50):
  Needs: ceil(50/16) = 4 blocks
  block_table[0] = [block_5, block_12, block_7, block_3]

Request 1 (sequence length 33):
  Needs: ceil(33/16) = 3 blocks
  block_table[1] = [block_1, block_9, block_15]

Physical KV Cache:
  [block_0, block_1, ..., block_15]
  Each block stores: [K: [block_size, num_heads, head_dim],
                       V: [block_size, num_heads, head_dim]]
```

**Decode时的KV Cache访问**：
```python
# 伪代码
def paged_attention_decode(
    query,        # [num_reqs, num_heads, head_dim]
    key_cache,    # [num_blocks, block_size, num_heads, head_dim]
    value_cache,  # [num_blocks, block_size, num_heads, head_dim]
    block_table,  # [num_reqs, max_blocks_per_seq]
    seq_lens,     # [num_reqs]
):
    for req_idx in range(num_reqs):
        q = query[req_idx]                    # [num_heads, head_dim]
        seq_len = seq_lens[req_idx]          # scalar
        blocks = block_table[req_idx]         # [num_blocks_for_this_req]
        
        # 从多个blocks中gather K和V
        k = gather_kv_from_blocks(key_cache, blocks, seq_len)
        v = gather_kv_from_blocks(value_cache, blocks, seq_len)
        # k, v shape: [seq_len, num_heads, head_dim]
        
        # 做attention
        output[req_idx] = attention(q, k, v)
```

---

## 性能优化技巧

### 1. Continuous Batching
- 不等待整个batch完成，新request可以随时加入
- Request完成后立即移除，释放资源

### 2. Chunked Prefill
- 长prompts分chunk处理
- 避免一个长prompt阻塞整个batch
- `max_num_batched_tokens` 控制chunk大小

### 3. CUDA Graph
- 对于固定大小的decode batch，使用CUDA graph
- 减少kernel launch开销
- 提升decode throughput

### 4. Prefix Caching
- 相同prefix的requests共享KV cache blocks
- 减少重复计算

---

## 总结

**整个流程的本质**：

1. **用户视角**：
   - 提供多个prompts
   - 等待得到完整的生成结果

2. **vLLM内部**：
   - 将prompts tokenize后变成requests
   - 每个iteration调度一部分tokens
   - Prefill阶段处理prompts，Decode阶段逐个生成tokens
   - 通过continuous batching和paged attention优化资源利用
   - 最终所有requests完成后返回结果

3. **关键创新**：
   - **Paged Attention**: 灵活高效的KV cache管理
   - **Continuous Batching**: 动态batch，无需等待
   - **Chunked Prefill**: 分块处理长prompts
   - **Mixed Batch**: prefill和decode可以混合在一起

**代码调用链总结**：
```
LLM.generate()
  → _validate_and_add_requests()
    → _add_request() × N
      → Processor.process_inputs() (tokenize)
      → LLMEngine.add_request()
        → OutputProcessor.add_request()
        → EngineCore.add_request()
  → _run_engine()
    → while has_unfinished_requests():
      → LLMEngine.step()
        → EngineCore.get_output()
          → EngineCore.step()
            → Scheduler.schedule()          # 决定处理哪些tokens
            → GPUModelRunner.execute_model()
              → _prepare_inputs()          # 准备tensor
              → _model_forward()           # 模型计算
              → _sample()                  # 采样
            → Scheduler.update_from_output()  # 更新状态
        → OutputProcessor.process_outputs()   # detokenize
```

希望这个详细的分析能帮助你理解vLLM的batch推理流程！

