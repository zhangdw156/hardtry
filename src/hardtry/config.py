from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HuggingFace 模型路径"})
    use_4bit_quant: bool = field(default=True, metadata={"help": "是否使用 4-bit 量化"})
    attn_implementation: str = field(default="sdpa", metadata={"help": "Attention 实现: eager, sdpa, flash_attention_2"})

@dataclass
class ScriptArguments:
    # --- 数据控制 ---
    data_path: str = field(default="../../data/bfcl_multi_turn.json")
    train_subset_size: int = field(default=5000, metadata={"help": "训练集采样数量"})
    validation_split_percentage: float = field(default=0.05, metadata={"help": "验证集比例"})
    
    # --- LoRA/DoRA 控制 ---
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    use_dora: bool = field(default=True, metadata={"help": "是否开启 DoRA"})
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # --- 其他 ---
    total_batch_size: int = field(default=64, metadata={"help": "目标总 Batch Size (自动计算梯度累积)"})
    system_prompt_save_path: Optional[str] = field(default=None)
    prompt_demo_save_path: Optional[str] = field(default=None)