import os
import torch
from transformers import (
    Trainer, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import swanlab
from datasets import load_dataset

class BaseHardTryTrainer(Trainer):
    def __init__(self, model_args, training_args, script_args):
        self.local_rank = training_args.local_rank
        if self.local_rank == -1:
            self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.model_args = model_args
        self.script_args = script_args
        
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        auto_grad_accum = self.script_args.total_batch_size // (
            training_args.per_device_train_batch_size * world_size
        )
        training_args.gradient_accumulation_steps = max(1, auto_grad_accum)

        self.training_args = training_args

        if self.local_rank<=0:
            self._init_swanlab()

        model, tokenizer = self._init_model_and_tokenizer()
        
        train_dataset, eval_dataset = self._process_dataset(tokenizer)

        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)
        )

    def _init_swanlab(self):
        """
        收集所有参数并初始化 SwanLab 实验
        """
        # 1. 收集参数：将三个来源的参数全部转为字典
        # training_args 是 HF 对象，自带 to_dict()
        # model_args 和 script_args 是 dataclass，用 asdict()
        try:
            all_config = {
                "script_args": asdict(self.script_args),
                "model_config": asdict(self.model_args),
                "training_args": self.training_args.to_dict()
            }
        except Exception as e:
            print(f"Warning: Failed to serialize args for SwanLab: {e}")
            all_config = {}

        # 2. 获取实验名和项目名
        # 假设 script_args 里定义了 swanlab_project 和 swanlab_experiment
        # 使用 getattr 提供默认值，防止报错
        project_name = getattr(self.script_args, "swanlab_project", "hardtry_default")
        experiment_name = getattr(self.script_args, "swanlab_experiment", None)

        print(f"🚀 Initializing SwanLab: Project={project_name}, Experiment={experiment_name}")

        # 3. 初始化
        swanlab.init(
            project=project_name,
            experiment_name=experiment_name,
            config=all_config
        )

    def _init_model_and_tokenizer(self):
        """内部方法：负责加载模型、量化和 LoRA"""
        # A. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        tokenizer.padding_side = "right"

        # 获取微调类型
        tune_type = self.script_args.tune_type.lower()
        print(f"🔥 Training Strategy: {tune_type.upper()}")

        # =====================================================
        # B. Quantization 配置
        # =====================================================
        bnb_config = None
        if tune_type != "full" and self.model_args.use_4bit_quant:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # C. Model
        is_deepspeed_enabled = self.training_args.deepspeed is not None
        device_map = None if is_deepspeed_enabled else {"": self.local_rank}
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if tune_type == "full" else "auto",
            device_map=device_map,
            attn_implementation=self.model_args.attn_implementation,
        )

        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

        if tune_type == "full":
            model.config.use_cache = False
            pass
            
        elif tune_type in ["lora", "dora"]:
            # 决定是否开启 DoRA
            is_dora = (tune_type == "dora")
            
            peft_config = LoraConfig(
                r=self.script_args.lora_r,
                lora_alpha=self.script_args.lora_alpha,
                lora_dropout=self.script_args.lora_dropout,
                task_type="CAUSAL_LM",
                target_modules=self.script_args.target_modules,
                use_dora=is_dora,
                bias="none"
            )
            model = get_peft_model(model, peft_config)
            model.config.use_cache = False
        
        else:
            raise ValueError(f"Unsupported tune_type: {tune_type}. Use 'full', 'lora', or 'dora'.")
        
        return model, tokenizer

    def _process_dataset(self, tokenizer):
        # 加载
        raw_ds = load_dataset("json", data_files=self.script_args.data_path, split="train")
        if self.script_args.train_subset_size > 0:
            raw_ds = raw_ds.shuffle(seed=42).select(range(min(len(raw_ds), self.script_args.train_subset_size)))
        
        # 保存 System Prompt Demo
        if self.local_rank <= 0 and self.script_args.system_prompt_save_path:
            temp_res = self._convert_to_openai(raw_ds[0])
            if temp_res["messages"]:
                with open(self.script_args.system_prompt_save_path, "w", encoding="utf-8") as f:
                    f.write(temp_res["messages"][0]["content"])

        # 转换格式
        converted_ds = raw_ds.map(self._convert_to_openai, remove_columns=raw_ds.column_names)
        
        # 保存 Completeion Demo
        if self.local_rank <= 0 and self.script_args.prompt_demo_save_path:
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
    
    def _convert_to_openai(self, example):
        """
        转换成openai格式的messages
        """
        raise NotImplementedError("必须手动实现数据格式转换函数")