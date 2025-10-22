# vLLM 模型推理完整流程分析

## 概述

vLLM 是一个高性能的大语言模型推理引擎，采用了异步架构、智能调度和高效的内存管理（PagedAttention）。本文档详细分析从发送请求到生成最终输出的完整流程。

## 整体架构

vLLM 采用分层架构设计：

```
┌─────────────────────────────────────────────────────────────┐
│                      API Server Layer                        │
│              (OpenAI-Compatible API Server)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Engine Client Layer                       │
│                  (AsyncLLM / LLM Class)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Input Processor                            │
│           (Tokenization & Multi-Modal)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Engine Core                               │
│        (Scheduler + Model Executor + KV Cache)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Output Processor                            │
│            (Detokenization & Formatting)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
                  Response
```

## 详细流程

### 第 1 阶段：请求接收与解析

**入口点：** `vllm/entrypoints/openai/api_server.py`

1. **HTTP 请求接收**
   - 用户通过 OpenAI 兼容的 API 发送请求
   - 请求格式：`POST /v1/completions` 或 `/v1/chat/completions`
   - 请求参数包含：prompt、model、sampling parameters 等

2. **请求验证与解析**
   - 验证请求格式和参数
   - 创建 `SamplingParams` 对象（temperature, top_p, max_tokens 等）
   - 生成唯一的 `request_id`

**关键代码位置：**
- `vllm/entrypoints/openai/serving_completion.py`
- `vllm/entrypoints/openai/serving_chat.py`

---

### 第 2 阶段：输入处理 (Input Processing)

**处理器：** `vllm/v1/engine/processor.py`

1. **文本 Tokenization**
   - 使用 HuggingFace tokenizer 将输入文本转换为 token IDs
   - 处理特殊 tokens (BOS, EOS, padding 等)
   - 支持 truncation（如果超过 max_model_len）

2. **多模态数据处理**
   - 处理图像、视频、音频等多模态输入
   - 将多模态数据转换为模型可用的 embeddings
   - 管理多模态缓存

3. **创建 EngineCoreRequest**
   - 封装所有请求信息：
     - `request_id`: 唯一标识符
     - `prompt_token_ids`: token 序列
     - `sampling_params`: 采样参数
     - `arrival_time`: 请求到达时间
     - `lora_request`: LoRA 配置（如果使用）
     - `multi_modal_data`: 多模态数据

**关键类：**
```python
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    arrival_time: float
    ...
```

---

### 第 3 阶段：请求添加到引擎 (Add Request to Engine)

**引擎层：** `vllm/v1/engine/async_llm.py` 或 `vllm/v1/engine/llm_engine.py`

1. **AsyncLLM.generate() 方法**
   - 为每个请求创建 `RequestOutputCollector`（用于异步输出）
   - 处理 n > 1 的情况（多个候选输出）
   - 将请求添加到 OutputProcessor

2. **OutputProcessor.add_request()**
   - 创建 `RequestState` 对象跟踪请求状态
   - 初始化 detokenizer 状态
   - 设置统计信息（arrival_time, metrics 等）

3. **EngineCore.add_request()**
   - 将请求添加到 EngineCore 的输入队列
   - 如果使用多进程，通过 ZMQ 发送到 EngineCore 进程

**关键流程：**
```
API Server
    ↓
AsyncLLM.generate()
    ↓
OutputProcessor.add_request()  (创建 RequestState)
    ↓
EngineCore.add_request()       (加入队列)
```

---

### 第 4 阶段：调度循环 (Scheduling Loop)

**核心引擎：** `vllm/v1/engine/core.py`

#### 4.1 主循环 (Busy Loop)

EngineCore 运行一个持续的循环：

```python
def run_busy_loop(self):
    while True:
        # 1) 处理输入队列（新请求）
        self._process_input_queue()
        
        # 2) 执行引擎步骤
        self._process_engine_step()
```

#### 4.2 单步执行 (Single Step)

```python
def step(self) -> tuple[EngineCoreOutputs, bool]:
    # 1. 调度：决定处理哪些请求
    scheduler_output = self.scheduler.schedule()
    
    # 2. 执行：运行模型前向传播
    model_output = self.model_executor.execute_model(scheduler_output)
    
    # 3. 更新：从模型输出更新调度器状态
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    
    return engine_core_outputs, model_executed
```

---

### 第 5 阶段：智能调度 (Intelligent Scheduling)

**调度器：** `vllm/v1/core/sched/scheduler.py`

#### 5.1 调度策略

调度器负责决定每个 iteration 处理哪些请求和多少 tokens：

1. **请求队列管理**
   - **WAITING 队列**: 新到达的请求
   - **RUNNING 队列**: 正在处理的请求
   - 支持优先级调度（FCFS 或 Priority）

2. **调度决策**
   ```python
   def schedule(self) -> SchedulerOutput:
       # 计算可用资源
       token_budget = max_num_batched_tokens
       
       # 1) 优先处理 RUNNING 请求（续写）
       for request in running_queue:
           num_tokens_to_schedule = calculate_tokens(request)
           if can_allocate_kv_cache(request, num_tokens_to_schedule):
               scheduled_requests.append(request)
               token_budget -= num_tokens_to_schedule
       
       # 2) 处理 WAITING 请求（新请求）
       for request in waiting_queue:
           if token_budget > 0 and can_allocate_kv_cache(request):
               scheduled_requests.append(request)
               token_budget -= request.num_prompt_tokens
       
       return SchedulerOutput(scheduled_requests, ...)
   ```

3. **关键特性**
   - **Chunked Prefill**: 大 prompt 可以分块处理
   - **Prefix Caching**: 共享相同前缀的请求可以复用 KV cache
   - **Preemption**: 资源不足时可以抢占低优先级请求
   - **Mixed Batching**: 同一批次可以包含 prefill 和 decode 请求

#### 5.2 SchedulerOutput 结构

```python
class SchedulerOutput:
    scheduled_requests: List[Request]
    num_scheduled_tokens: Dict[str, int]  # 每个请求调度的 token 数
    req_to_new_blocks: Dict[str, List[Block]]  # 新分配的 KV cache 块
    input_ids: torch.Tensor  # 所有请求的 input token IDs
    positions: torch.Tensor  # Token positions
    ...
```

---

### 第 6 阶段：KV Cache 管理

**KV Cache 管理器：** `vllm/v1/core/kv_cache_manager.py`

#### 6.1 PagedAttention 核心机制

vLLM 的关键创新：将 KV cache 按块（block）管理，类似操作系统的分页内存：

1. **Block Pool**
   - 预先分配固定数量的内存块
   - 每个 block 存储固定数量 tokens 的 KV cache（通常 16 或 32）
   - 动态分配和回收

2. **分配策略**
   ```python
   def allocate_slots(request, num_new_tokens):
       # 1. 检查 prefix cache（已缓存的前缀）
       cached_blocks = find_cached_blocks(request)
       
       # 2. 计算需要的新 blocks
       num_blocks_needed = ceil(num_new_tokens / block_size)
       
       # 3. 从 block pool 分配
       new_blocks = block_pool.allocate(num_blocks_needed)
       
       # 4. 更新请求的 block table
       request.block_table.extend(new_blocks)
       
       return new_blocks
   ```

3. **Prefix Caching**
   - 使用 hash 表存储已计算的 KV cache
   - 多个请求可以共享相同的前缀块
   - 大幅减少重复计算

4. **回收机制**
   - 请求完成后释放 blocks
   - LRU 策略淘汰不常用的缓存
   - 支持 swap to CPU（实验性）

---

### 第 7 阶段：模型执行 (Model Execution)

**模型运行器：** `vllm/v1/worker/gpu_model_runner.py`

#### 7.1 输入准备

```python
def execute_model(scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
    # 1. 准备输入张量
    input_ids = scheduler_output.input_ids
    positions = scheduler_output.positions
    
    # 2. 构建 Attention Metadata
    attn_metadata = prepare_attention_metadata(
        block_tables=scheduler_output.block_tables,
        context_lens=scheduler_output.context_lens,
        ...
    )
    
    # 3. 准备多模态输入
    mm_inputs = prepare_multimodal_inputs(...)
```ca

#### 7.2 前向传播

```python
    # 4. 执行模型前向传播
    with torch.inference_mode():
        hidden_states = model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            **mm_inputs
        )
    
    # 5. 计算 logits
    logits = model.compute_logits(hidden_states)
```

#### 7.3 优化技术

- **CUDA Graphs**: 减少 kernel launch 开销
- **Continuous Batching**: 动态批处理，无需等待所有序列完成
- **Tensor Parallelism**: 跨 GPU 分割模型
- **Pipeline Parallelism**: 多 GPU 流水线执行

---

### 第 8 阶段：Token 采样 (Token Sampling)

**采样器：** `vllm/v1/sample/sampler.py`

#### 8.1 采样流程

```python
class Sampler:
    def forward(self, logits, sampling_metadata) -> SamplerOutput:
        # 1. 计算原始 logprobs（如果需要）
        if need_logprobs:
            raw_logprobs = compute_logprobs(logits)
        
        # 2. 转换为 float32
        logits = logits.float()
        
        # 3. 应用 allowed/banned token IDs
        apply_token_filters(logits, sampling_metadata)
        
        # 4. 应用 logit bias
        apply_logit_bias(logits, sampling_metadata)
        
        # 5. 应用惩罚 (repetition, frequency, presence)
        apply_penalties(logits, output_tokens, sampling_metadata)
        
        # 6. 采样
        sampled_tokens = self.sample(logits, sampling_metadata)
        
        return SamplerOutput(sampled_tokens, logprobs, ...)
```

#### 8.2 采样策略

```python
def sample(logits, sampling_metadata):
    # Greedy sampling (temperature = 0)
    if all_greedy:
        return logits.argmax(dim=-1)
    
    # Random sampling
    # 1. 应用 temperature
    logits = logits / temperature
    
    # 2. 应用 min_p
    apply_min_p(logits, min_p)
    
    # 3. 应用 top_k / top_p
    logits = apply_top_k_top_p(logits, top_k, top_p)
    
    # 4. 从概率分布采样
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    
    return sampled
```

#### 8.3 采样参数

- **temperature**: 控制随机性（0 = greedy, >1 = 更随机）
- **top_p**: nucleus sampling，累积概率阈值
- **top_k**: 只考虑概率最高的 k 个 tokens
- **repetition_penalty**: 惩罚重复 tokens
- **frequency_penalty**: 基于频率的惩罚
- **presence_penalty**: 是否出现过的惩罚

---

### 第 9 阶段：输出处理 (Output Processing)

**输出处理器：** `vllm/v1/engine/output_processor.py`

#### 9.1 处理流程

```python
def process_outputs(engine_core_outputs) -> OutputProcessorOutput:
    for output in engine_core_outputs:
        request_id = output.request_id
        req_state = self.request_states[request_id]
        
        # 1. Detokenization (增量解码)
        new_text = self.detokenizer.decode_incremental(
            new_token_ids=output.new_token_ids,
            prev_tokens=req_state.output_token_ids
        )
        
        # 2. 检测停止条件
        finish_reason = self.check_stop_conditions(
            output, req_state, new_text
        )
        
        # 3. 创建 CompletionOutput
        completion_output = CompletionOutput(
            index=req_state.index,
            text=new_text,
            token_ids=output.new_token_ids,
            cumulative_logprob=output.cumulative_logprob,
            logprobs=output.logprobs,
            finish_reason=finish_reason,
        )
        
        # 4. 创建 RequestOutput
        request_output = RequestOutput(
            request_id=request_id,
            prompt=req_state.prompt,
            prompt_token_ids=req_state.prompt_token_ids,
            outputs=[completion_output],
            finished=(finish_reason is not None),
            metrics=compute_metrics(req_state),
        )
        
        # 5. 放入输出队列或返回列表
        if req_state.queue:  # AsyncLLM
            req_state.queue.put_nowait(request_output)
        else:  # LLMEngine
            request_outputs.append(request_output)
        
        # 6. 清理已完成的请求
        if finish_reason:
            self.request_states.pop(request_id)
            abort_requests.append(request_id)
    
    return OutputProcessorOutput(request_outputs, abort_requests)
```

#### 9.2 停止条件检测

```python
def check_stop_conditions(output, req_state, new_text):
    # 1. 达到 max_tokens
    if len(req_state.output_token_ids) >= sampling_params.max_tokens:
        return "length"
    
    # 2. 遇到 EOS token
    if output.new_token_ids[-1] == eos_token_id:
        return "stop"
    
    # 3. 匹配 stop strings
    for stop_str in sampling_params.stop:
        if stop_str in new_text:
            return "stop"
    
    return None  # 继续生成
```

#### 9.3 增量 Detokenization

vLLM 使用增量解码优化：
- 不重新解码所有 tokens
- 只解码新生成的 tokens
- 处理多字节 UTF-8 字符
- 处理 tokenizer 的特殊行为（如空格处理）

---

### 第 10 阶段：结果返回 (Response)

#### 10.1 流式输出 (Streaming)

对于流式请求：

```python
async def generate(prompt, sampling_params, request_id):
    # 获取异步生成器
    result_generator = engine.generate(prompt, sampling_params, request_id)
    
    # 流式返回每个输出
    async for request_output in result_generator:
        if not request_output.finished:
            # 增量输出
            yield format_chunk(request_output)
        else:
            # 最终输出
            yield format_final(request_output)
```

**优点：**
- 降低首 token 延迟（Time to First Token, TTFT）
- 改善用户体验
- 支持 Server-Sent Events (SSE)

#### 10.2 非流式输出

对于非流式请求：
- 收集所有输出直到完成
- 返回完整的 response
- 包含完整的 metrics

#### 10.3 输出格式

**RequestOutput 结构：**
```python
class RequestOutput:
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]
    finished: bool
    metrics: RequestMetrics  # 延迟、吞吐量等统计
    
class CompletionOutput:
    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: float
    logprobs: Dict  # top logprobs
    finish_reason: str  # "length", "stop", "abort"
```

---

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户发送请求                              │
│                    POST /v1/completions                          │
│              {prompt, max_tokens, temperature, ...}              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第1步：请求接收与解析                            │
│                  (OpenAI API Server)                             │
│  • 验证请求参数                                                   │
│  • 创建 SamplingParams                                           │
│  • 生成 request_id                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第2步：输入处理                                  │
│                  (Processor)                                     │
│  • Tokenization: text → token_ids                               │
│  • 多模态处理: image/video → embeddings                          │
│  • 创建 EngineCoreRequest                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第3步：添加请求到引擎                            │
│                  (AsyncLLM / LLMEngine)                          │
│  • 创建 RequestState (OutputProcessor)                          │
│  • 创建异步输出队列 (RequestOutputCollector)                     │
│  • 添加到 EngineCore                                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第4步：引擎核心循环                              │
│                  (EngineCore.run_busy_loop)                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  While True:                                             │  │
│  │    1. 处理输入队列（接收新请求）                           │  │
│  │    2. 执行单步 (step)：                                   │  │
│  │       ├─ 调度 (Scheduler.schedule)                       │  │
│  │       ├─ 执行模型 (ModelExecutor.execute_model)          │  │
│  │       └─ 更新状态 (Scheduler.update_from_output)         │  │
│  │    3. 输出结果到输出队列                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第5步：智能调度                                  │
│                  (Scheduler.schedule)                            │
│                                                                  │
│  输入：WAITING 队列 + RUNNING 队列                               │
│        ↓                                                         │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ 调度决策：                                              │    │
│  │ 1. 计算可用 token budget                               │    │
│  │ 2. 优先续写 RUNNING 请求 (decode)                      │    │
│  │ 3. 调度 WAITING 请求 (prefill)                         │    │
│  │ 4. 检查 KV cache 可用性                                │    │
│  │ 5. 支持 chunked prefill & prefix caching              │    │
│  └────────────────────────────────────────────────────────┘    │
│        ↓                                                         │
│  输出：SchedulerOutput                                           │
│    • scheduled_requests                                         │
│    • num_scheduled_tokens                                       │
│    • block_tables                                               │
│    • input_ids, positions                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第6步：KV Cache 管理                            │
│                  (KVCacheManager)                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ PagedAttention 机制：                                   │    │
│  │                                                         │    │
│  │ Block Pool (预分配的内存块)                            │    │
│  │  ┌─────┐┌─────┐┌─────┐┌─────┐                         │    │
│  │  │Block││Block││Block││Block│ ...                      │    │
│  │  │  0  ││  1  ││  2  ││  3  │                         │    │
│  │  └─────┘└─────┘└─────┘└─────┘                         │    │
│  │                                                         │    │
│  │ 请求 A: Block [0, 1, 5]                                │    │
│  │ 请求 B: Block [0, 1, 7]  ← 共享前缀 [0,1]             │    │
│  │ 请求 C: Block [2, 3, 4]                                │    │
│  │                                                         │    │
│  │ 功能：                                                   │    │
│  │ • 动态分配/释放 blocks                                  │    │
│  │ • Prefix caching (共享相同前缀)                        │    │
│  │ • Copy-on-write (多候选生成)                           │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第7步：模型执行                                  │
│                  (GPUModelRunner)                                │
│                                                                  │
│  1. 准备输入                                                     │
│     ├─ input_ids: [batch_size, seq_len]                        │
│     ├─ positions: [batch_size, seq_len]                        │
│     └─ attn_metadata (block_tables, context_lens, ...)         │
│                                                                  │
│  2. 模型前向传播                                                 │
│     ┌────────────────────────────────────────────────────┐     │
│     │  with torch.inference_mode():                      │     │
│     │    hidden_states = model(                          │     │
│     │      input_ids=input_ids,                          │     │
│     │      positions=positions,                          │     │
│     │      kv_caches=kv_caches,  ← 使用 PagedAttention   │     │
│     │      attn_metadata=attn_metadata                   │     │
│     │    )                                                │     │
│     │    logits = model.lm_head(hidden_states)           │     │
│     └────────────────────────────────────────────────────┘     │
│                                                                  │
│  3. 优化技术                                                     │
│     • CUDA Graphs (减少 kernel launch 开销)                     │
│     • Continuous Batching (动态批处理)                          │
│     • Tensor Parallelism (跨 GPU)                              │
│     • FlashAttention (高效注意力计算)                           │
│                                                                  │
│  输出：ModelRunnerOutput                                         │
│    • sampled_token_ids                                          │
│    • logprobs                                                   │
│    • hidden_states (如果需要)                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第8步：Token 采样                               │
│                  (Sampler)                                       │
│                                                                  │
│  输入：logits [batch_size, vocab_size]                          │
│        ↓                                                         │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ 采样流程：                                              │    │
│  │                                                         │    │
│  │ 1. 预处理                                               │    │
│  │    ├─ 转换为 float32                                   │    │
│  │    ├─ 应用 allowed/banned token IDs                    │    │
│  │    └─ 应用 logit bias                                  │    │
│  │                                                         │    │
│  │ 2. 应用惩罚                                             │    │
│  │    ├─ repetition_penalty                               │    │
│  │    ├─ frequency_penalty                                │    │
│  │    └─ presence_penalty                                 │    │
│  │                                                         │    │
│  │ 3. 采样策略                                             │    │
│  │    ┌──────────────────────────────────────────┐       │    │
│  │    │ if temperature == 0:                     │       │    │
│  │    │   # Greedy sampling                      │       │    │
│  │    │   token = logits.argmax()                │       │    │
│  │    │ else:                                     │       │    │
│  │    │   # Random sampling                      │       │    │
│  │    │   logits = logits / temperature          │       │    │
│  │    │   logits = apply_min_p(logits)           │       │    │
│  │    │   logits = apply_top_k_top_p(logits)     │       │    │
│  │    │   probs = softmax(logits)                │       │    │
│  │    │   token = multinomial(probs)             │       │    │
│  │    └──────────────────────────────────────────┘       │    │
│  │                                                         │    │
│  │ 4. 收集 logprobs (如果需要)                            │    │
│  └────────────────────────────────────────────────────────┘    │
│        ↓                                                         │
│  输出：SamplerOutput                                             │
│    • sampled_token_ids                                          │
│    • logprobs                                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第9步：输出处理                                  │
│                  (OutputProcessor)                               │
│                                                                  │
│  对每个 EngineCoreOutput:                                        │
│                                                                  │
│  1. 增量 Detokenization                                         │
│     ┌────────────────────────────────────────────────────┐     │
│     │ new_text = detokenizer.decode_incremental(         │     │
│     │     new_token_ids=[新生成的 token],                │     │
│     │     prev_tokens=[之前生成的 tokens]                │     │
│     │ )                                                   │     │
│     └────────────────────────────────────────────────────┘     │
│                                                                  │
│  2. 检测停止条件                                                 │
│     • 达到 max_tokens？                                          │
│     • 遇到 EOS token？                                           │
│     • 匹配 stop strings？                                        │
│                                                                  │
│  3. 创建 RequestOutput                                          │
│     ┌────────────────────────────────────────────────────┐     │
│     │ RequestOutput(                                     │     │
│     │   request_id="...",                                │     │
│     │   prompt="用户输入",                               │     │
│     │   outputs=[                                        │     │
│     │     CompletionOutput(                              │     │
│     │       text="生成的文本",                           │     │
│     │       token_ids=[token序列],                       │     │
│     │       logprobs={...},                              │     │
│     │       finish_reason="stop"/"length"/None           │     │
│     │     )                                               │     │
│     │   ],                                                │     │
│     │   finished=True/False,                             │     │
│     │   metrics=RequestMetrics(...)                      │     │
│     │ )                                                   │     │
│     └────────────────────────────────────────────────────┘     │
│                                                                  │
│  4. 输出路由                                                     │
│     • AsyncLLM: 放入请求的异步队列                              │
│     • LLMEngine: 添加到返回列表                                 │
│                                                                  │
│  5. 清理完成的请求                                               │
│     • 从 request_states 删除                                    │
│     • 通知 EngineCore abort 请求                                │
│     • 释放 KV cache blocks                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第10步：结果返回                                 │
│                                                                  │
│  流式输出 (Streaming):                                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ async for request_output in result_generator:          │    │
│  │     if stream:                                          │    │
│  │         # SSE 格式流式返回                              │    │
│  │         chunk = {                                       │    │
│  │           "choices": [{                                 │    │
│  │             "delta": {"content": new_text},             │    │
│  │             "finish_reason": None/finish_reason         │    │
│  │           }]                                             │    │
│  │         }                                                │    │
│  │         yield f"data: {json.dumps(chunk)}\n\n"          │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  非流式输出:                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ # 等待完成                                              │    │
│  │ final_output = await collect_all_outputs()              │    │
│  │                                                         │    │
│  │ response = {                                            │    │
│  │   "choices": [{                                         │    │
│  │     "text": complete_generated_text,                    │    │
│  │     "finish_reason": "stop",                            │    │
│  │     "logprobs": {...}                                   │    │
│  │   }],                                                    │    │
│  │   "usage": {                                            │    │
│  │     "prompt_tokens": 10,                                │    │
│  │     "completion_tokens": 50,                            │    │
│  │     "total_tokens": 60                                  │    │
│  │   }                                                      │    │
│  │ }                                                        │    │
│  │ return JSONResponse(response)                           │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   用户收到响应    │
                    └──────────────────┘
```

---

## 核心组件详解

### 1. Scheduler（调度器）

**职责：**
- 管理请求队列（WAITING、RUNNING）
- 决定每个 step 执行哪些请求
- 分配计算资源（tokens、KV cache）
- 实现 preemption 和 priority

**关键算法：**
- **Continuous Batching**: 不等待所有序列完成，动态添加/移除请求
- **Chunked Prefill**: 大 prompt 分块处理，与 decode 混合批处理
- **Prefix Caching**: 识别和共享相同前缀

### 2. KVCacheManager（KV缓存管理器）

**职责：**
- 管理所有请求的 KV cache
- 实现 PagedAttention 的块分配
- 处理 prefix caching
- 支持 copy-on-write (beam search)

**PagedAttention 优势：**
- 减少内存碎片
- 提高内存利用率（接近 100%）
- 支持大 batch size
- 动态共享前缀

### 3. ModelExecutor（模型执行器）

**职责：**
- 管理 workers（多 GPU 场景）
- 执行模型前向传播
- 协调 tensor/pipeline parallelism
- 管理 CUDA graphs

**优化技术：**
- **FlashAttention**: 高效的注意力计算
- **Fused Kernels**: 融合多个操作减少内存访问
- **Quantization**: 支持 INT8/INT4 量化

### 4. OutputProcessor（输出处理器）

**职责：**
- 跟踪所有请求状态
- 增量 detokenization
- 检测停止条件
- 计算 metrics
- 路由输出到正确的队列

---

## 性能优化技巧

### 1. PagedAttention
- **问题**: 传统方法需要连续内存存储 KV cache
- **解决**: 分块存储，类似操作系统分页
- **收益**: 2-4x 吞吐量提升

### 2. Continuous Batching
- **问题**: 静态批处理等待所有序列完成
- **解决**: 动态添加/移除请求
- **收益**: 减少平均延迟，提高 GPU 利用率

### 3. Prefix Caching
- **问题**: 相同前缀重复计算
- **解决**: 缓存共享的 KV cache blocks
- **收益**: 多轮对话场景大幅加速

### 4. Chunked Prefill
- **问题**: 大 prompt 阻塞 decode 请求
- **解决**: 分块处理 prefill，与 decode 混合
- **收益**: 降低 TPOT (Time Per Output Token)

### 5. CUDA Graphs
- **问题**: Kernel launch 开销
- **解决**: 预录制 GPU 操作图
- **收益**: 降低延迟，特别是小 batch

---

## 关键指标

### 延迟指标

1. **TTFT (Time to First Token)**
   - 从请求到第一个 token 的时间
   - 受 prefill 时间影响
   - 优化：chunked prefill, prefix caching

2. **TPOT (Time Per Output Token)**
   - 每个输出 token 的平均时间
   - 受 decode 效率影响
   - 优化：batching, CUDA graphs

3. **E2E Latency (End-to-End Latency)**
   - 完整请求的总时间
   - TTFT + (num_tokens - 1) × TPOT

### 吞吐量指标

1. **Throughput (tokens/second)**
   - 系统整体吞吐量
   - 优化：大 batch, KV cache 复用

2. **QPS (Queries Per Second)**
   - 每秒处理的请求数
   - 优化：continuous batching

---

## 代码位置总结

| 组件 | 文件路径 |
|-----|---------|
| API Server | `vllm/entrypoints/openai/api_server.py` |
| LLM 类 | `vllm/entrypoints/llm.py` |
| AsyncLLM | `vllm/v1/engine/async_llm.py` |
| LLMEngine | `vllm/v1/engine/llm_engine.py` |
| EngineCore | `vllm/v1/engine/core.py` |
| Scheduler | `vllm/v1/core/sched/scheduler.py` |
| KVCacheManager | `vllm/v1/core/kv_cache_manager.py` |
| ModelRunner | `vllm/v1/worker/gpu_model_runner.py` |
| Sampler | `vllm/v1/sample/sampler.py` |
| OutputProcessor | `vllm/v1/engine/output_processor.py` |
| Processor | `vllm/v1/engine/processor.py` |

---

## 总结

vLLM 的推理流程可以概括为 **10 个关键步骤**：

1. **请求接收** → 解析 HTTP 请求
2. **输入处理** → Tokenization + 多模态处理
3. **添加请求** → 加入引擎队列
4. **引擎循环** → 持续处理请求
5. **智能调度** → 决定执行哪些请求
6. **KV Cache** → PagedAttention 分配内存
7. **模型执行** → 前向传播生成 logits
8. **Token 采样** → 从 logits 选择下一个 token
9. **输出处理** → Detokenization + 停止检测
10. **结果返回** → 流式或批量返回用户

**核心创新：**
- **PagedAttention**: 高效的 KV cache 管理
- **Continuous Batching**: 动态批处理
- **Prefix Caching**: 共享相同前缀
- **异步架构**: 高并发支持

这些技术使 vLLM 成为目前最快的开源 LLM 推理引擎之一！

