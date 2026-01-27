import sys
import os
import re
import json
import ast
from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments

from hardtry.config import ModelArguments, ScriptArguments
from hardtry.base_trainer import BaseHardTryTrainer

from typing import override
from hardtry.converter import convert_python_to_xml

class Full1Trainer(BaseHardTryTrainer):
    
    # ==========================================
    # 1. 私有辅助方法
    # ==========================================
    def _parse_assistant_content(self, content):
        if not content:
            return content
        
        delimiter = "</think>\n\n"
        
        if delimiter in content:
            parts = content.rsplit(delimiter, 1)
            prefix = parts[0] + delimiter 
            suffix = parts[1]
        else:
            prefix = ""
            suffix = content

        formatted_tools = convert_python_to_xml(suffix)

        if prefix:
            return prefix + formatted_tools
        else:
            return formatted_tools

    def _convert_to_openai(self, example):
        """HardGen -> OpenAI 格式转换"""
        messages = []

        # A. System Prompt
        if example.get("0") is not None:
            system_data = example["0"]
            
            tools_list = system_data.get("tools", [])
            if isinstance(tools_list, str):
                try:
                    tools_list = json.loads(tools_list)
                except json.JSONDecodeError:
                    try:
                        tools_list = ast.literal_eval(tools_list)
                    except Exception as e:
                        if self.local_rank == 0:
                            print(f"[Warning] Failed to parse tools: {e}")
                        tools_list = []
            new_instruction = (
                "You are an expert in composing functions. You are given a question and a set of possible functions. "
                "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n"
                "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\n"
                "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
                "Continue to output functions to call until you have fulfilled the user's request to the best of your ability."
            )

            if tools_list:
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
                system_content = new_instruction + tools_header
            else:
                # 如果没有 tools，仅保留指令（视情况而定）
                system_content = new_instruction
                
            messages.append({"role": "system", "content": new_instruction + tools_header})

        # B. Dialog Turns
        turn_keys = sorted([k for k in example.keys() if k.isdigit() and k != "0"], key=int)
        for key in turn_keys:
            turn_data = example[key]
            if not turn_data: continue
            role, content = turn_data.get("role"), turn_data.get("content", "")

            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": self._parse_assistant_content(content)})
            elif role == "tool":
                try:
                    tool_list = ast.literal_eval(content) if isinstance(content, str) else content
                except: tool_list = []
                for call_result in tool_list:
                    for call_signature, result_val in call_result.items():
                        func_name = call_signature.split('(')[0]
                        try: final_res = json.loads(result_val) if isinstance(result_val, str) else result_val
                        except: final_res = result_val
                        tool_json = {"name": func_name, "arguments": final_res}
                        messages.append({"role": "tool", "content": json.dumps(tool_json, ensure_ascii=False)})

        return {"messages": messages}

    def _format_and_mask(self, example, tokenizer):
        """Tokenize & Masking"""
        messages = example["messages"]
        if messages[-1]["role"] != "assistant":
            return {"input_ids": [], "labels": [], "attention_mask": []}
        
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        context_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=False)
        
        full_enc = tokenizer(full_text, add_special_tokens=False)
        context_enc = tokenizer(context_text, add_special_tokens=False)
        
        input_ids = full_enc["input_ids"]
        labels = list(input_ids)
        context_len = len(context_enc["input_ids"])
        
        # Mask
        if context_len > len(input_ids):
            labels[:] = [-100] * len(labels)
        else:
            labels[:context_len] = [-100] * context_len
            
        return {"input_ids": input_ids, "labels": labels, "attention_mask": full_enc["attention_mask"]}

    # ==========================================
    # 2. 实现抽象方法 process_dataset
    # ==========================================
    @override
    def process_dataset(self, tokenizer):
        # 加载
        raw_ds = load_dataset("json", data_files=self.script_args.data_path, split="train")
        if self.script_args.train_subset_size > 0:
            raw_ds = raw_ds.shuffle(seed=42).select(range(min(len(raw_ds), self.script_args.train_subset_size)))
        
        # 保存 System Prompt Demo
        if self.local_rank == 0 and self.script_args.system_prompt_save_path:
            temp_res = self._convert_to_openai(raw_ds[0])
            if temp_res["messages"]:
                with open(self.script_args.system_prompt_save_path, "w", encoding="utf-8") as f:
                    f.write(temp_res["messages"][0]["content"])

        # 转换格式
        converted_ds = raw_ds.map(self._convert_to_openai, remove_columns=raw_ds.column_names)
        
        # 保存 Completeion Demo
        if self.local_rank == 0 and self.script_args.prompt_demo_save_path:
            try:
                prompt_demo = tokenizer.apply_chat_template(
                    converted_ds[0]["messages"], add_generation_prompt=False, tokenize=False
                )
                with open(self.script_args.prompt_demo_save_path, "w", encoding="utf-8") as f:
                    f.write(prompt_demo)
            except Exception as e:
                print(f"Warning: Failed to save prompt demo: {e}")

        # Tokenize & Mask
        processed_dataset = converted_ds.map(
            lambda x: self._format_and_mask(x, tokenizer), 
            remove_columns=converted_ds.column_names
        ).filter(lambda x: len(x["input_ids"]) > 0)

        # 验证集切分
        split_ds = processed_dataset.train_test_split(
            test_size=self.script_args.validation_split_percentage, 
            seed=42
        )
        return split_ds["train"], split_ds["test"]

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments, ScriptArguments))
    
    # 自动查找 yaml
    yaml_file = None
    for arg in sys.argv:
        if arg.endswith(".yaml"):
            yaml_file = arg
            break
            
    if yaml_file:
        model_args, training_args, script_args = parser.parse_yaml_file(yaml_file=yaml_file)
    else:
        print("Error: No config.yaml provided.")
        exit(1)

    os.environ["SWANLAB_PROJECT"] = "hardtry"

    # 初始化并开始训练
    trainer = Full1Trainer(model_args, training_args, script_args)
    trainer.train()