import json 
import math

with open('cache_vllm_inference_output copy.json', 'r') as f:
    vllm_data = json.load(f)

with open('cache_fsdp_output.json', 'r') as f:
    fsdp_data = json.load(f)

vllm_logprobs = vllm_data['response']['tokens']

fsdp_data = fsdp_data['results']
fsdp_logprobs = []
for result in fsdp_data:
    if result['index'] >= 129:
        fsdp_logprobs.append(result)


print(len(vllm_logprobs))
print(len(fsdp_logprobs))

c = 0
for i in range(len(vllm_logprobs)):
    vllm_prob = math.exp(vllm_logprobs[i]['logprob'])
    fsdp_prob = fsdp_logprobs[i]['prob']
    is_ = fsdp_prob / vllm_prob
    print(f"token id: {vllm_logprobs[i]['token_id']}, vllm prob: {vllm_prob}, fsdp prob: {fsdp_prob}, is: {is_}")

    if is_ > 0.9 and is_ < 1.1:
        c += 1
print(f"total: {c}")