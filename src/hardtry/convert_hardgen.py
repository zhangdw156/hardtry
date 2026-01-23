import json
from datasets import load_dataset

# 1. 设置文件路径
input_path = "data/bfcl_multi_turn.json"
output_path = "data/bfcl_messages_format.json"

# 2. 使用 load_dataset 加载数据
print(f"正在加载数据集: {input_path}")
ds = load_dataset("json", data_files=input_path, split="train")

def process_hardgen_to_messages(example):
    """
    将 HardGen 格式的单条样本转换为 OpenAI messages 格式
    """
    messages = []
    
    # --- A. 处理 System Prompt (Key "0") ---
    # dataset 加载后，key 对应的值如果是 None 表示该样本没有此轮（因为是稀疏列）
    if example.get("0") is not None:
        # load_dataset 加载后的 nested json 有时已经是 dict，有时是 string
        # 根据你的原始文件，这里已经是结构化的 dict (包含 content, role, tools 等)
        system_data = example["0"]
        system_content = system_data.get("content", "")
        
        # System 包含 content (及 tools 定义)
        messages.append({
            "role": "system",
            "content": system_content
        })

    # --- B. 处理对话轮次 (Key "1", "2", "3"...) ---
    # 获取所有是数字的 key，排除 "0"
    all_keys = list(example.keys())
    turn_keys = [k for k in all_keys if k.isdigit() and k != "0"]
    
    # 必须按数字大小排序，保证对话顺序
    sorted_keys = sorted(turn_keys, key=int)
    
    for key in sorted_keys:
        turn_data = example[key]
        
        # 因为 dataset 是稀疏的，某些样本可能没有 key "10"，值为 None
        if turn_data is None:
            continue
            
        role = turn_data.get("role")
        content = turn_data.get("content")
        
        # --- C. 映射逻辑 ---
        if role == "user":
            messages.append({"role": "user", "content": content})
            
        elif role == "assistant":
            # 保留 <think> 和 function call
            messages.append({"role": "assistant", "content": content})
            
        elif role == "tool":
            messages.append({"role": "tool", "content": content})

    # 返回新的格式
    return {"messages": messages}

# 3. 执行 map 处理
print("正在转换数据格式...")
# remove_columns 很重要，因为我们要把原来杂乱的 "0", "1"... 列都删掉，只留 "messages"
converted_ds = ds.map(
    process_hardgen_to_messages,
    remove_columns=ds.column_names
)

# 4. 打印一条样本检查
print("\n转换后样本示例 (第1条):")
print(json.dumps(converted_ds[0]["messages"][:3], indent=2, ensure_ascii=False))
print("...")

# 5. 保存为最终 JSON 文件 (用于 Swift/LLaMA-Factory)
final_data = converted_ds.to_list()
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print("完成！")
