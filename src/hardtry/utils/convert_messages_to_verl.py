import os
import sys
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

# Hydra 仅用于入口装饰，config 用 OmegaConf 合并
try:
    import hydra
except ImportError:
    hydra = None

_CONF_DIR = Path(__file__).resolve().parent / "conf"


def _ensure_config_file_in_argv():
    """兼容 run.py 传入的单个 yaml 路径：转为 config_file=path，便于 Hydra 解析。"""
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")) and "=" not in sys.argv[1]:
        sys.argv[1] = f"config_file={sys.argv[1]}"


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


def run(cfg):
    """实际执行逻辑，接收 OmegaConf 配置。"""
    # 若指定了 config_file，则合并其内容（config_file 自身字段不覆盖已有 CLI 覆盖）
    if cfg.get("config_file"):
        user = OmegaConf.load(cfg.config_file)
        # 合并：cfg 为底，user 覆盖，避免 config_file 路径写回
        merged = OmegaConf.merge(cfg, user)
        merged.config_file = cfg.get("config_file")
        cfg = merged

    OmegaConf.resolve(cfg)
    input_path = cfg.get("input_path") or ""
    output_dir = cfg.get("output_dir") or ""
    model_path = cfg.get("model_path") or ""
    if not input_path or not output_dir or not model_path:
        raise ValueError(
            "请在配置文件或命令行中指定 input_path、output_dir、model_path。"
        )

    test_size = cfg.get("test_size", 0.05)
    seed = cfg.get("seed", 42)
    num_proc = cfg.get("num_proc", 8)
    data_source = cfg.get("data_source", "sloop")
    ability = cfg.get("ability", "function_call")
    max_samples = cfg.get("max_samples")

    print(f"开始加载数据集: {input_path}")
    ds = load_dataset("json", data_files=input_path, split="train")

    # 保留条数限制
    if max_samples is not None and max_samples > 0:
        n = min(int(max_samples), len(ds))
        ds = ds.select(range(n))
        print(f"已限制为前 {n} 条数据（max_samples={max_samples}）")

    # 1. 划分数据集
    split_ds = ds.train_test_split(test_size=test_size, seed=seed)

    # 2. 转换格式
    processed_datasets = {}
    for key in ["train", "test"]:
        print(f"正在处理 {key} 分片...")
        processed_datasets[key] = split_ds[key].map(
            make_map_fn(key, data_source, ability),
            with_indices=True,
            num_proc=num_proc,
        )

    # 3. 统计并打印信息
    print("\n加载 Tokenizer 进行统计...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for key in ["train", "test"]:
        p_max, g_max = get_stats(processed_datasets[key], tokenizer)
        print("\n" + "=" * 40)
        print(f"【{key.upper()}】分片统计结果")
        print(f"数量: {len(processed_datasets[key])}")
        print(f"Max Prompt Tokens: {p_max}")
        print(f"Max GT Tokens: {g_max}")
        print("=" * 40)

    # 4. 保存文件
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    processed_datasets["train"].to_parquet(train_path)
    processed_datasets["test"].to_parquet(test_path)
    print(f"\n保存完成！\n训练集: {train_path}\n测试集: {test_path}")


_config_path = str(_CONF_DIR.resolve())

if hydra is not None:

    @hydra.main(
        config_path=_config_path,
        config_name="convert_messages_to_verl",
        version_base="1.3",
    )
    def main(cfg):
        run(cfg)
else:

    def main(_cfg=None):
        raise ImportError("请安装 hydra-core 以使用 Hydra 配置：pip install hydra-core")


if __name__ == "__main__":
    _ensure_config_file_in_argv()
    if not os.path.isdir(_CONF_DIR):
        raise FileNotFoundError(f"默认配置目录不存在: {_CONF_DIR}")
    main()
