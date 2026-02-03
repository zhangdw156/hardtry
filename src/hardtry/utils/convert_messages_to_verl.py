import os
import sys
from dataclasses import dataclass, field

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class DataArguments:
    input_path: str = field(metadata={"help": "原始 OpenAI 格式 JSON 路径"})
    output_dir: str = field(metadata={"help": "输出 parquet 文件的目录"})
    model_path: str = field(metadata={"help": "用于统计长度的 Tokenizer 路径"})
    test_size: float = field(default=0.05, metadata={"help": "验证集比例"})
    seed: int = field(default=42, metadata={"help": "随机种子"})
    num_proc: int = field(default=8, metadata={"help": "并行进程数"})
    data_source: str = field(default="sloop", metadata={"help": "数据来源标识"})
    ability: str = field(default="function_call", metadata={"help": "能力类型标签"})


def make_map_fn(split, data_source, ability):
    def process_fn(example, idx):
        messages = example.pop("messages")
        return {
            "data_source": data_source,
            "prompt": messages[:-1],
            "ability": ability,
            "reward_model": {"style": "rule", "ground_truth": messages[-1]["content"]},
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }

    return process_fn


def get_stats(dataset, tokenizer):
    max_prompt_len = 0
    max_gt_len = 0

    for row in tqdm(dataset, desc="统计 Token 长度"):
        # 应用模板计算 Prompt 长度
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        max_prompt_len = max(max_prompt_len, len(prompt_token_ids))

        # 计算 Ground Truth 长度
        gt_text = row["reward_model"]["ground_truth"]
        gt_token_ids = tokenizer.encode(gt_text, add_special_tokens=False)
        max_gt_len = max(max_gt_len, len(gt_token_ids))

    return max_prompt_len, max_gt_len


def main():
    parser = HfArgumentParser(DataArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # 如果只传了一个 yaml 文件路径
        script_args = parser.parse_yaml_file(yaml_file=sys.argv[1])[0]
    else:
        # 否则按常规命令行参数解析
        script_args = parser.parse_args_into_dataclasses()[0]

    print(f"开始加载数据集: {script_args.input_path}")
    ds = load_dataset("json", data_files=script_args.input_path, split="train")

    # 1. 划分数据集
    split_ds = ds.train_test_split(
        test_size=script_args.test_size, seed=script_args.seed
    )

    # 2. 转换格式
    processed_datasets = {}
    for key in ["train", "test"]:
        print(f"正在处理 {key} 分片...")
        processed_datasets[key] = split_ds[key].map(
            make_map_fn(key, script_args.data_source, script_args.ability),
            with_indices=True,
            num_proc=script_args.num_proc,
        )

    # 3. 统计并打印信息
    print("\n加载 Tokenizer 进行统计...")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)

    for key in ["train", "test"]:
        p_max, g_max = get_stats(processed_datasets[key], tokenizer)
        print("\n" + "=" * 40)
        print(f"【{key.upper()}】分片统计结果")
        print(f"数量: {len(processed_datasets[key])}")
        print(f"Max Prompt Tokens: {p_max}")
        print(f"Max GT Tokens: {g_max}")
        print("=" * 40)

    # 4. 保存文件
    os.makedirs(script_args.output_dir, exist_ok=True)
    train_path = os.path.join(script_args.output_dir, "train.parquet")
    test_path = os.path.join(script_args.output_dir, "test.parquet")

    processed_datasets["train"].to_parquet(train_path)
    processed_datasets["test"].to_parquet(test_path)
    print(f"\n保存完成！\n训练集: {train_path}\n测试集: {test_path}")


if __name__ == "__main__":
    main()
