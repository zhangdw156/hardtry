import os
import re
import json
import ast
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

# ==========================================
# 1. 配置与参数
# ==========================================
@dataclass
class ScriptArguments:
    total_batch_size: int = field(default=64)
    per_device_batch_size: int = field(default=4)
    model_path: str = field(default="/dfs/data/models/Qwen3-4B-Thinking-2507")
    local_rank: int = field(default=int(os.environ.get("LOCAL_RANK", -1)))
    deepspeed: str = field(default=None)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# 动态计算梯度累积
world_size = int(os.environ.get("WORLD_SIZE", 1))
grad_accum_steps = max(1, script_args.total_batch_size // (script_args.per_device_batch_size * world_size))

# ==========================================
# 2. 数据处理逻辑
# ==========================================

def parse_assistant_content(content):
    """将 [func(a=1)] 转换为 <tool_call> JSON 格式"""
    if not content or '[' not in content:
        return content

    match = re.search(r'\[(.*)\]\s*$', content.strip(), re.DOTALL)
    if not match:
        return content

    full_call_str = match.group(0)
    inner_content = match.group(1)
    
    # 增强版正则：支持 key="val", key=123, key=True
    calls = re.findall(r'(\w+)\((.*?)\)', inner_content)
    
    tool_calls_formatted = []
    for name, args_str in calls:
        args_dict = {}
        if args_str.strip():
            # 匹配 key = value，支持多种值类型
            arg_pairs = re.findall(r'(\w+)\s*=\s*(("[^"]*")|(\'[^\']*\')|([^,]+))', args_str)
            for pair in arg_pairs:
                k = pair[0]
                v = pair[1].strip().strip("'\"")
                try:
                    if '.' in v: args_dict[k] = float(v)
                    else: args_dict[k] = int(v)
                except ValueError:
                    if v.lower() == "true": args_dict[k] = True
                    elif v.lower() == "false": args_dict[k] = False
                    else: args_dict[k] = v
        
        call_obj = {"name": name, "arguments": args_dict}
        tool_calls_formatted.append(f"<tool_call>\n{json.dumps(call_obj, ensure_ascii=False)}\n</tool_call>")

    prefix = content[:content.rfind(full_call_str)].strip()
    return f"{prefix}\n\n" + "\n".join(tool_calls_formatted) if prefix else "\n".join(tool_calls_formatted)

def convert_to_openai(example):
    """构造全新的 System Prompt 并处理多轮对话"""
    messages = []

    # --- A. 构造全新 System Prompt ---
    if example.get("0") is not None:
        system_data = example["0"]
        # 参考旧逻辑的指令
        new_instruction = (
            "You are an expert in composing functions. You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n"
            "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\n"
            "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
            "Continue to output functions to call until you have fulfilled the user's request to the best of your ability."
        )
        
        # 动态获取并格式化 Tools
        tools_list = system_data.get("tools", [])
        if isinstance(tools_list, str):
            try: tools_list = ast.literal_eval(tools_list)
            except: tools_list = []

        tools_header = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        for tool in tools_list:
            tools_header += f"\n{json.dumps(tool, ensure_ascii=False)}"
        
        tools_header += (
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )
        messages.append({"role": "system", "content": new_instruction + tools_header})

    # --- B. 对话轮次处理 ---
    turn_keys = sorted([k for k in example.keys() if k.isdigit() and k != "0"], key=int)
    for key in turn_keys:
        turn_data = example[key]
        if not turn_data: continue
        role, content = turn_data.get("role"), turn_data.get("content", "")

        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": parse_assistant_content(content)})
        elif role == "tool":
            # 解决反斜杠转义问题
            try:
                tool_list = ast.literal_eval(content) if isinstance(content, str) else content
            except: tool_list = []
            for call_result in tool_list:
                for call_signature, result_val in call_result.items():
                    func_name = call_signature.split('(')[0]
                    try:
                        final_res = json.loads(result_val) if isinstance(result_val, str) else result_val
                    except: final_res = result_val
                    
                    tool_json = {"name": func_name, "arguments": final_res}
                    messages.append({"role": "tool", "content": json.dumps(tool_json, ensure_ascii=False)})

    return {"messages": messages}

# ==========================================
# 3. 训练核心流程
# ==========================================

# 加载并转换数据
raw_ds = load_dataset("json", data_files="../../data/bfcl_multi_turn.json", split="train")
raw_ds = raw_ds.shuffle(seed=42).select(range(min(5000, len(raw_ds))))
converted_ds = raw_ds.map(convert_to_openai, remove_columns=raw_ds.column_names)

# 保存 Prompt 样本
if script_args.local_rank <= 0:
    with open("../../data/system_prompt.txt", "w", encoding="utf-8") as f:
        f.write(converted_ds[0]["messages"][0]["content"])

# Tokenizer 准备
processor = AutoTokenizer.from_pretrained(script_args.model_path)
processor.padding_side = "right" 

def format_and_mask(example):
    messages = example["messages"]
    if messages[-1]["role"] != "assistant":
        return {"input_ids": [], "labels": [], "attention_mask": []}
    
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    context_text = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=False)
    
    full_enc = processor(full_text, add_special_tokens=False)
    context_enc = processor(context_text, add_special_tokens=False)
    
    input_ids = full_enc["input_ids"]
    labels = list(input_ids)
    context_len = len(context_enc["input_ids"])
    
    # 对 Prompt 部分进行 Mask
    labels[:context_len] = [-100] * context_len
    return {"input_ids": input_ids, "labels": labels, "attention_mask": full_enc["attention_mask"]}

processed_dataset = converted_ds.map(format_and_mask, remove_columns=converted_ds.column_names).filter(lambda x: len(x["input_ids"]) > 0)

# 模型加载 (4-bit & DDP 兼容)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_path,
    quantization_config=bnb_config,
    device_map={"": max(0, script_args.local_rank)},
    attn_implementation="sdpa",
)

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_dora=True
)
model = get_peft_model(model, peft_config)
model.config.use_cache = False

# 训练参数
training_args = TrainingArguments(
    output_dir=f"../../checkpoints/{os.path.basename(os.getcwd())}",
    per_device_train_batch_size=script_args.per_device_batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    num_train_epochs=1,
    learning_rate=1e-4,
    bf16=True,
    optim="paged_adamw_32bit",
    report_to="swanlab" if script_args.local_rank <= 0 else "none",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    save_strategy="epoch",
    deepspeed=script_args.deepspeed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=DataCollatorForSeq2Seq(processor, padding=True, pad_to_multiple_of=8)
)

trainer.train()