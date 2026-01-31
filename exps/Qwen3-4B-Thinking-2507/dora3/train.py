import sys
import os
import json
import torch
from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments
from typing import override
from hardtry.config import ModelArguments, ScriptArguments
from hardtry.base_trainer import BaseHardTryTrainer

class Dora3Trainer(BaseHardTryTrainer):
    
    @override
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
                messages.append({"role": "user", "content": content})

        return {"messages": messages}

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

    # 实例化并运行
    trainer = Dora3Trainer(model_args, training_args, script_args)
    trainer.train()