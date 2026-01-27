import os
import sys
import torch
import logging
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser
)
from peft import PeftModel

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MergeArguments:
    base_model_path: str = field(metadata={"help": "基座模型路径"})
    lora_path: str = field(metadata={"help": "LoRA/DoRA 权重路径"})
    output_path: str = field(metadata={"help": "合并后模型的保存路径"})
    device: str = field(default="auto", metadata={"help": "加载设备 (auto, cpu, cuda)"})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "加载精度 (float16, bfloat16, float32)"})

class ModelMerger:
    def __init__(self, args: MergeArguments):
        self.args = args
        self.dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }

    def _get_dtype(self):
        return self.dtype_map.get(self.args.torch_dtype, torch.bfloat16)

    def run(self):
        logger.info(f"Step 1: Loading base model from {self.args.base_model_path}...")
        
        # 1. 加载基座模型 (必须以非量化形式加载)
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model_path,
                torch_dtype=self._get_dtype(),
                device_map=self.args.device,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise e

        logger.info(f"Step 2: Loading LoRA adapter from {self.args.lora_path}...")
        
        # 2. 加载 Adapter
        try:
            model = PeftModel.from_pretrained(base_model, self.args.lora_path)
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")
            raise e

        logger.info("Step 3: Merging weights (merge_and_unload)...")
        
        # 3. 合并权重
        # 这一步会将 LoRA/DoRA 权重计算并加回基座模型参数中，变成一个标准的 Dense 模型
        merged_model = model.merge_and_unload()

        # 确保输出目录存在
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)

        logger.info(f"Step 4: Saving merged model to {self.args.output_path}...")
        merged_model.save_pretrained(
            self.args.output_path, 
            safe_serialization=True # 推荐使用 safetensors 格式
        )

        logger.info("Step 5: Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.base_model_path, 
            trust_remote_code=True
        )
        tokenizer.save_pretrained(self.args.output_path)

        logger.info("✅ Merge Complete! You can now serve the merged model.")

if __name__ == "__main__":
    parser = HfArgumentParser((MergeArguments,))
    
    # 检查命令行参数，支持 .yaml 文件
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        merge_args, = parser.parse_yaml_file(yaml_file=sys.argv[1])
    else:
        # 也支持直接命令行传参
        merge_args, = parser.parse_args_into_dataclasses()

    merger = ModelMerger(merge_args)
    merger.run()