import os
import sys
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from dataclasses import dataclass, field

# --- 参数定义 ---
@dataclass
class ScriptArguments:
    total_batch_size: int = field(default=64, metadata={"help": "目标总Batch Size"})
    per_device_batch_size: int = field(default=4, metadata={"help": "单张卡的显存允许的Batch Size"})
    model_path: str = field(default="/dfs/data/models/Qwen3-4B-Thinking-2507", metadata={"help": "模型路径"})
    max_seq_length: int = field(default=4096, metadata={"help": "强制截断长度"})
    local_rank: int = field(default=-1, metadata={"help": "DeepSpeed会自动传入"})
    # 【新增】定义 deepspeed 参数，防止 parser 报错
    deepspeed: str = field(default=None, metadata={"help": "DeepSpeed 配置文件路径"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# --- 1. 动态计算梯度累积步数 ---
# 获取当前使用的 GPU 数量 (World Size)
world_size = int(os.environ.get("WORLD_SIZE", 1))

# 计算公式: Total = Per_Device * World_Size * Grad_Accum
# 所以: Grad_Accum = Total / (Per_Device * World_Size)
grad_accum_steps = script_args.total_batch_size // (script_args.per_device_batch_size * world_size)

if grad_accum_steps < 1:
    grad_accum_steps = 1
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"警告: 总BatchSize设置过小，已强制将梯度累积设为1。实际总BatchSize为: {script_args.per_device_batch_size * world_size}")

if int(os.environ.get("RANK", 0)) == 0:
    print(f"--- 训练配置 ---")
    print(f"GPU数量 (World Size): {world_size}")
    print(f"目标总 Batch Size: {script_args.total_batch_size}")
    print(f"单卡 Batch Size: {script_args.per_device_batch_size}")
    print(f"计算得出的梯度累积步数: {grad_accum_steps}")
    print(f"----------------")

# --- 2. 数据处理 ---
# 加载 Tokenizer
processor = AutoTokenizer.from_pretrained(script_args.model_path)
processor.padding_side = "right" # 训练必须右填充

raw_data_path = "../data/bfcl_multi_turn.json"
raw_ds = load_dataset("json", data_files=raw_data_path, split="train")
# 采样并打乱
raw_ds = raw_ds.shuffle(seed=42).select(range(5000))

def convert_to_openai(example):
    # (保留你原来的转换逻辑，此处省略以节省篇幅，逻辑不变)
    messages = []
    if example.get("0") is not None:
        messages.append({"role": "system", "content": example["0"].get("content", "")})
    
    all_keys = list(example.keys())
    turn_keys = sorted([k for k in all_keys if k.isdigit() and k != "0"], key=int)
    
    for key in turn_keys:
        turn_data = example[key]
        if turn_data:
            messages.append({"role": turn_data.get("role"), "content": turn_data.get("content")})
    return {"messages": messages}

converted_ds = raw_ds.map(convert_to_openai, remove_columns=raw_ds.column_names)

def format_and_mask_last_turn(example):
    messages = example["messages"]
    
    # 上下文检查
    if messages[-1]["role"] != "assistant":
        return {"input_ids": [], "labels": [], "attention_mask": []}

    # 1. 应用模板
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    context_text = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=False)
    
    # 2. Tokenize
    full_tokens = processor(
        full_text, 
        return_tensors="pt", 
        add_special_tokens=False,
        truncation=True
    )
    
    input_ids = full_tokens["input_ids"][0]
    attention_mask = full_tokens["attention_mask"][0]
    labels = input_ids.clone()
    
    # 3. Masking
    context_tokens = processor(context_text, return_tensors="pt", add_special_tokens=False)
    context_len = context_tokens["input_ids"].shape[1]
    
    if context_len > len(input_ids):
        labels[:] = -100
    else:
        labels[:context_len] = -100
        
    return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist(), "labels": labels.tolist()}

processed_dataset = converted_ds.map(format_and_mask_last_turn, remove_columns=converted_ds.column_names)
# 过滤掉无效数据
processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)

# --- 3. 模型加载 ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 【关键】多卡环境下，必须指定 device_map 为当前进程的 local_rank
# DeepSpeed 会管理 local_rank 环境变量
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_map = {"": local_rank}

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_path,
    quantization_config=bnb_config,
    device_map=device_map, # 显式指定 GPU
    # attn_implementation="flash_attention_2",
    attn_implementation="sdpa",
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_dora=True,
)

model = get_peft_model(model, peft_config)
model.config.use_cache = False # 训练时必须关闭 cache

if local_rank == 0:
    model.print_trainable_parameters()

# --- 4. 训练设置 ---
# SwanLab 配置 (通过环境变量控制)
if local_rank == 0:
    import os
    os.environ["SWANLAB_PROJECT"] = "hardtry"

training_args = TrainingArguments(
    output_dir="../checkpoints",
    per_device_train_batch_size=script_args.per_device_batch_size,
    gradient_accumulation_steps=grad_accum_steps, # <--- 动态传入
    num_train_epochs=1,
    learning_rate=1e-4,
    logging_steps=10,
    fp16=False,
    bf16=True,
    optim="paged_adamw_32bit",
    report_to="swanlab" if local_rank == 0 else "none", # 只有主进程汇报
    gradient_checkpointing=True,
    group_by_length=True,
    ddp_find_unused_parameters=False, # 这通常能减少 DDP 报错
    save_strategy="epoch",
    deepspeed=script_args.deepspeed
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor,
    padding=True,
    pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
)

trainer.train()