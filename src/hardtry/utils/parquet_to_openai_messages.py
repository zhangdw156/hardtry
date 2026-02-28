"""
从 convert_messages_to_verl 产出的 parquet（train.parquet / test.parquet）还原为 OpenAI 格式的 messages，
便于与 verl 使用同一份数据做 SFT 等。还原规则：messages = prompt + [{"role": "assistant", "content": ground_truth}]。
最后一条的 role 在 parquet 中未保存，默认按 "assistant" 还原。
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def _row_to_messages(row):
    prompt = row["prompt"]
    if not isinstance(prompt, list):
        prompt = list(prompt) if prompt else []
    gt = row.get("reward_model") or {}
    if not isinstance(gt, dict):
        gt = dict(gt) if hasattr(gt, "items") else {}
    ground_truth = gt.get("ground_truth", "") or ""
    return list(prompt) + [{"role": "assistant", "content": ground_truth}]


def load_parquet(path):
    ds = load_dataset("parquet", data_files=str(path), split="train")
    return [ds[i] for i in range(len(ds))]


def _write_messages_to_file(list_of_messages, out_path, fmt="jsonl"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for messages in list_of_messages:
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"messages": m} for m in list_of_messages],
                f,
                ensure_ascii=False,
                indent=2,
            )
    print(f"  已写入 {len(list_of_messages)} 条 -> {out_path}")


def run(
    input_dir,
    train_path=None,
    test_path=None,
    output_path=None,
    output_dir=None,
    output_format=None,
):
    """
    input_dir: 目录，内含 train.parquet、test.parquet；或仅用 train_path/test_path 指定文件。
    output_path: 单文件输出路径（.json 或 .jsonl），train+test 合并。
    output_dir: 若指定，则按 train/test 分别写出，与 RL 完全同源同划分（SFT 训练用 train，评估用 test）。
    output_format: 若为 "jsonl" 则按行写；否则写为一个 JSON 数组。
    """
    if train_path is None and test_path is None:
        input_dir = Path(input_dir)
        train_path = input_dir / "train.parquet"
        test_path = input_dir / "test.parquet"
    else:
        train_path = Path(train_path) if train_path else None
        test_path = Path(test_path) if test_path else None
        input_dir = Path(input_dir) if input_dir else (train_path.parent if train_path else Path("."))

    train_rows = load_parquet(str(train_path)) if train_path and train_path.exists() else []
    test_rows = load_parquet(str(test_path)) if test_path and test_path.exists() else []

    if not train_rows and not test_rows:
        raise FileNotFoundError(
            "未找到 parquet 数据，请指定 input_dir 或 train_path/test_path。"
        )

    fmt = (output_format or "jsonl").lower()

    # 按 train / test 分别输出：train.parquet → train.jsonl，test.parquet → test.jsonl（与 RL 同划分，使用时分开指定）
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if train_rows:
            train_messages = [_row_to_messages(r) for r in train_rows]
            _write_messages_to_file(
                train_messages,
                out_dir / "train.jsonl",
                fmt=fmt,
            )
        if test_rows:
            test_messages = [_row_to_messages(r) for r in test_rows]
            _write_messages_to_file(
                test_messages,
                out_dir / "test.jsonl",
                fmt=fmt,
            )
        return str(out_dir)

    # 单文件输出（train 在前，test 在后）
    rows = train_rows + test_rows
    list_of_messages = [_row_to_messages(r) for r in rows]
    out = Path(output_path) if output_path else Path(input_dir) / "openai_messages.jsonl"
    _write_messages_to_file(list_of_messages, out, fmt=fmt)
    return str(out)


def main():
    ap = argparse.ArgumentParser(
        description="从 hardgen_1k 等 parquet 还原为 OpenAI messages 格式（供 SFT 等使用）。"
    )
    ap.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="含 train.parquet / test.parquet 的目录（如 data/hardgen_1k）",
    )
    ap.add_argument("--train_path", type=str, default=None, help="train.parquet 路径")
    ap.add_argument("--test_path", type=str, default=None, help="test.parquet 路径")
    ap.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="单文件输出路径（train+test 合并）；与 --output_dir 二选一",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="按 train/test 分别输出到该目录：train.parquet→train.jsonl，test.parquet→test.jsonl，与 RL 同划分，SFT 中分开指定",
    )
    ap.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="jsonl",
        help="输出格式，默认 jsonl",
    )
    args = ap.parse_args()
    run(
        input_dir=args.input_dir or ".",
        train_path=args.train_path,
        test_path=args.test_path,
        output_path=args.output,
        output_dir=args.output_dir,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
