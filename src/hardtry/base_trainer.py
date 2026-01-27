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

class BaseHardTryTrainer(Trainer):
    def __init__(self, model_args, training_args, script_args):
        self.local_rank = training_args.local_rank
        if self.local_rank == -1:
            self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        # 保存参数，供内部方法使用
        self.model_args = model_args
        self.script_args = script_args
        # 注意：training_args 会在 super().__init__ 中被父类保存为 self.args
        
        # 1. 加载模型和 Tokenizer (这是我们在调用父类初始化前必须准备好的)
        model, tokenizer = self._init_model_and_tokenizer()
        
        # 2. 数据处理 (调用抽象方法，由子类实现)
        train_dataset, eval_dataset = self.process_dataset(tokenizer)
        
        # 3. 动态计算梯度累积
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        auto_grad_accum = self.script_args.total_batch_size // (
            training_args.per_device_train_batch_size * world_size
        )
        training_args.gradient_accumulation_steps = max(1, auto_grad_accum)

        # 4. 调用父类 Trainer 的初始化
        # 此时我们将准备好的 model, dataset, data_collator 全部传进去
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer, # 新版 transformers 建议传 processing_class (即 tokenizer)
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)
        )

    def _init_model_and_tokenizer(self):
        """内部方法：负责加载模型、量化和 LoRA"""
        # A. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        tokenizer.padding_side = "right"

        # B. Quantization
        bnb_config = None
        if self.model_args.use_4bit_quant:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # C. Model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
            attn_implementation=self.model_args.attn_implementation,
        )

        if self.model_args.use_4bit_quant:
            model = prepare_model_for_kbit_training(model)

        # D. LoRA
        peft_config = LoraConfig(
            r=self.script_args.lora_r,
            lora_alpha=self.script_args.lora_alpha,
            lora_dropout=self.script_args.lora_dropout,
            task_type="CAUSAL_LM",
            target_modules=self.script_args.target_modules,
            use_dora=self.script_args.use_dora,
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        model.config.use_cache = False
        
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            model.print_trainable_parameters()
            
        return model, tokenizer

    def process_dataset(self, tokenizer):
        """
        【抽象方法】必须由子类实现
        Return: (train_dataset, eval_dataset)
        """
        raise NotImplementedError("你必须在实验子类中实现 process_dataset 方法！")