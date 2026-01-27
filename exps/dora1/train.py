import sys
import os
import json
import torch
from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments

from hardtry.config import ModelArguments, ScriptArguments
from hardtry.base_trainer import BaseHardTryTrainer

class Dora1Trainer(BaseHardTryTrainer):
    
    def _convert_to_openai(self, example):
        """
        将 HardGen 格式的单条样本转换为 OpenAI messages 格式
        """
        messages = []

        # --- A. 处理 System Prompt (Key "0") ---
        if example.get("0") is not None:
            system_data = example["0"]
            system_content = system_data.get("content", "")
            messages.append({"role": "system", "content": system_content})

        # --- B. 处理对话轮次 (Key "1", "2", "3"...) ---
        all_keys = list(example.keys())
        turn_keys = [k for k in all_keys if k.isdigit() and k != "0"]
        sorted_keys = sorted(turn_keys, key=int)

        for key in sorted_keys:
            turn_data = example[key]
            if turn_data is None:
                continue

            role = turn_data.get("role")
            content = turn_data.get("content")

            # --- C. 映射逻辑 ---
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                messages.append({"role": "tool", "content": content})

        return {"messages": messages}

    def _format_and_mask_last_turn(self, example, tokenizer):
        """
        应用模板并只保留最后一次回复的 Loss
        """
        messages = example["messages"]

        # 1. 完整对话
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 2. 上下文检查
        if messages[-1]["role"] != "assistant":
            return {"input_ids": [], "labels": [], "attention_mask": []}

        context_messages = messages[:-1]
        context_text = tokenizer.apply_chat_template(
            context_messages, tokenize=False, add_generation_prompt=False
        )

        # 3. Tokenize
        full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        context_tokens = tokenizer(context_text, return_tensors="pt", add_special_tokens=False)

        input_ids = full_tokens["input_ids"][0]
        attention_mask = full_tokens["attention_mask"][0]

        # 4. 构造 Labels 并进行 Masking
        labels = input_ids.clone()
        context_len = context_tokens["input_ids"].shape[1]

        if context_len > len(input_ids):
            # 异常保护
            labels[:] = -100
        else:
            labels[:context_len] = -100

        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "labels": labels.tolist(),
        }

    def process_dataset(self, tokenizer):
        """
        [重写基类方法] 执行具体的数据加载与处理流程
        """
        # 1. 加载数据
        raw_ds = load_dataset("json", data_files=self.script_args.data_path, split="train")
        
        # 2. 采样与打乱
        # 使用 min 确保不会因为数据量不足报错
        select_num = min(len(raw_ds), self.script_args.train_subset_size)
        raw_ds = raw_ds.shuffle(seed=42).select(range(select_num))
        
        # 3. 保存原始 System Prompt
        # 只在主进程执行，避免多进程写文件冲突
        if self.local_rank == 0 and len(raw_ds) > 0:
            sys_prompt_path = self.script_args.system_prompt_save_path or "../../data/system_prompt.txt"
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(sys_prompt_path)), exist_ok=True)
            
            try:
                content = raw_ds[0]["0"]["content"].split('[{"name"')[0]
                with open(sys_prompt_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"System prompt saved to {sys_prompt_path}")
            except Exception as e:
                print(f"Warning: Failed to save system prompt: {e}")

        # 4. 格式转换 (HardGen -> OpenAI Messages)
        converted_ds = raw_ds.map(self._convert_to_openai, remove_columns=raw_ds.column_names)
        
        # 5. 保存 Prompt Demo (查看 Chat Template 效果)
        if self.local_rank == 0 and len(converted_ds) > 0:
            demo_path = self.script_args.prompt_demo_save_path or "../../data/prompt_demo.txt"
            try:
                prompt_demo = tokenizer.apply_chat_template(
                    converted_ds[0]["messages"], add_generation_prompt=False, tokenize=False
                )
                with open(demo_path, "w", encoding="utf-8") as f:
                    f.write(prompt_demo)
            except Exception as e:
                print(f"Warning: Failed to save prompt demo: {e}")

        # 6. Tokenize & Masking
        processed_dataset = converted_ds.map(
            lambda x: self._format_and_mask_last_turn(x, tokenizer),
            remove_columns=converted_ds.column_names
        )
        
        # 7. 过滤无效数据
        processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)
        
        # 8. 打印最大长度信息 (仅主进程)
        if self.local_rank == 0:
            lengths = [len(x) for x in processed_dataset["input_ids"]]
            if lengths:
                print(f"数据集中最长的序列长度是: {max(lengths)}")

        # 9. 切分验证集
        split_ds = processed_dataset.train_test_split(
            test_size=self.script_args.validation_split_percentage, 
            seed=42
        )
        
        return split_ds["train"], split_ds["test"]

# =================================================================
# 3. 主程序入口
# =================================================================
if __name__ == "__main__":
    # 解析 YAML 参数
    parser = HfArgumentParser((ModelArguments, TrainingArguments, ScriptArguments))
    
    # 1. 自动寻找以 .yaml 结尾的参数
    yaml_file = None
    for arg in sys.argv:
        if arg.endswith(".yaml"):
            yaml_file = arg
            break
            
    # 2. 如果找到了 yaml 文件就解析
    if yaml_file is not None:
        model_args, training_args, script_args = parser.parse_yaml_file(yaml_file=yaml_file)
    else:
        print("Error: No config.yaml file found in arguments.")
        exit(1)

    # 设置环境变量
    os.environ["SWANLAB_PROJECT"] = "hardtry"
    
    # 实例化并运行
    trainer = Dora1Trainer(model_args, training_args, script_args)
    trainer.train()