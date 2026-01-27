import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 配置路径
base_model_path = "/dfs/data/models/Qwen3-4B-Thinking-2507"
lora_path = "/dfs/data/work/hardtry/checkpoints/dora2/checkpoint-75"
# 合并后的新模型保存位置
output_path = "/dfs/data/models/sloop-4b_dora2"

print(f"Loading base model from {base_model_path}...")
# 注意：合并权重时建议用 float16 或 bfloat16，不要用 quantization (如 load_in_4bit)，否则无法无损合并
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, # A100 用 bfloat16 最好
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading LoRA adapter from {lora_path}...")
model = PeftModel.from_pretrained(base_model, lora_path)

print("Merging weights...")
# 这一步会将 DoRA 的权重计算并加到 Base Model 的参数里，同时移除 LoRA 模块
model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
model.save_pretrained(output_path)

# 顺便把 tokenizer 也存过去，方便直接用
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

print("Done! You can now serve the merged model.")