#!/usr/bin/env python3
"""
从实验的 eval5_results 目录读取 5 次 run 的 Base 指标，汇总并更新/追加到 exps/RESULT.csv。

用法:
  summarize_eval_to_result.py <实验名> [选项]
  summarize_eval_to_result.py verl17
  summarize_eval_to_result.py verl17 --eval-dir /path/to/eval5_results
  summarize_eval_to_result.py new_exp --framework verl --method EGPO --dataset "hardgen 14k" --notes "说明" --base-model "Qwen3-4B-Thinking-2507"

若 RESULT.csv 中已存在该实验名，则只更新 Mean、Result 1-5、Status、实验日期；
若不存在，则追加一行（需通过选项或 result_meta.yaml 提供 Framework/Method/Dataset/Notes/Base Model）。
可选：在实验目录下放置 result_meta.yaml，内容示例：
  framework: verl
  method: EGPO
  dataset: hardgen 14k
  notes: 4B-Thinking + EGPO ...
  base_model: Qwen3-4B-Thinking-2507
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from datetime import datetime


def find_repo_root() -> Path:
    """exps/commons/bin -> 仓库根目录（hardtry）。"""
    script = Path(__file__).resolve()
    # .../exps/commons/bin/summarize_eval_to_result.py -> .../exps
    exps = script.parent.parent.parent
    return exps.parent  # exps -> repo root


def collect_base_from_eval(eval_dir: Path) -> tuple[list[float], float]:
    """
    从 eval5_results 目录中读取 data_multi_turn_run_1..5_*.csv 的 Base 列（% 已去掉）。
    返回 ( [r1, r2, r3, r4, r5], mean )。
    """
    run_files: list[tuple[int, Path]] = []
    for f in eval_dir.iterdir():
        if not f.is_file() or f.suffix != ".csv":
            continue
        m = re.match(r"data_multi_turn_run_(\d+)_.*\.csv", f.name)
        if m:
            run_num = int(m.group(1))
            if 1 <= run_num <= 5:
                run_files.append((run_num, f))

    # 按 run 号去重（同一 run 多份时取其一），凑齐 1..5
    by_run = {r: p for r, p in run_files}
    if set(by_run.keys()) != {1, 2, 3, 4, 5}:
        raise SystemExit(
            f"错误: 在 {eval_dir} 中需要 data_multi_turn_run_1..5_*.csv 各一个，当前 run 号: {sorted(by_run.keys())!r}。"
        )
    run_files = [(i, by_run[i]) for i in range(1, 6)]
    values: list[float] = []

    for _, path in run_files:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        # 第一行表头，第二行数据；Base 是第 4 列（0-indexed 为 3）
        if len(lines) < 2:
            raise SystemExit(f"错误: {path} 行数不足。")
        header = lines[0].strip().split(",")
        try:
            base_idx = header.index("Base")
        except ValueError:
            raise SystemExit(f"错误: {path} 中未找到 Base 列，表头: {header}")
        data_line = lines[1].strip()
        # 简单按逗号分割可能因 Model 名含逗号而错位，用 csv 解析更稳
        reader = csv.reader([lines[0], lines[1]])
        next(reader)
        row = next(reader)
        if len(row) <= base_idx:
            raise SystemExit(f"错误: {path} 数据列数不足。")
        raw = row[base_idx].strip().replace("%", "")
        try:
            values.append(float(raw))
        except ValueError:
            raise SystemExit(f"错误: {path} Base 列无法解析为数字: {raw!r}")

    mean = round(sum(values) / len(values), 1)
    return values, mean


def load_result_csv(path: Path) -> list[list[str]]:
    """读取 RESULT.csv，返回每行作为 list（保留表头）。"""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def save_result_csv(path: Path, rows: list[list[str]]) -> None:
    """写回 RESULT.csv。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将实验 eval5_results 的 Base 汇总到 exps/RESULT.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    repo = find_repo_root()
    exps_dir = repo / "exps"
    default_result = exps_dir / "RESULT.csv"

    parser.add_argument("exp_name", help="实验名，如 verl17、full10")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help=f"eval5_results 目录，默认: exps/<实验名>/eval5_results",
    )
    parser.add_argument(
        "--result-csv",
        type=Path,
        default=default_result,
        help=f"RESULT.csv 路径，默认: {default_result}",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="实验日期，默认今天",
    )
    # 新增行时的元数据（若 RESULT 中已有该实验名则仅更新数值，这些可省略）
    parser.add_argument("--framework", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--base-model", default="", dest="base_model")
    args = parser.parse_args()

    eval_dir = args.eval_dir or (exps_dir / args.exp_name / "eval5_results")
    if not eval_dir.is_dir():
        print(f"错误: eval 目录不存在: {eval_dir}", file=sys.stderr)
        sys.exit(1)

    result_path = args.result_csv
    if not result_path.is_file():
        print(f"错误: RESULT 文件不存在: {result_path}", file=sys.stderr)
        sys.exit(1)

    # 可选：从实验目录的 result_meta.yaml 读取元数据（简单 key: value 解析）
    meta = {
        "framework": args.framework,
        "method": args.method,
        "dataset": args.dataset,
        "notes": args.notes,
        "base_model": args.base_model,
    }
    meta_path = exps_dir / args.exp_name / "result_meta.yaml"
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip().lower().replace("-", "_")
                    v = v.strip().strip('"\'')
                    if k in meta and not meta[k]:
                        meta[k] = v

    values, mean = collect_base_from_eval(eval_dir)
    # 与 RESULT 中一致：Mean 和 Result 1-5 用数字，可带一位小数
    r1, r2, r3, r4, r5 = values
    mean_str = str(mean)
    result_strs = [str(round(v, 1)) if v != int(v) else str(int(v)) for v in values]

    rows = load_result_csv(result_path)
    if len(rows) < 2:
        print("错误: RESULT.csv 至少需要表头与一行数据。", file=sys.stderr)
        sys.exit(1)

    header = rows[0]
    name_idx = header.index("Experiment Name")
    mean_idx = header.index("Mean (%)")
    r1_idx = header.index("Result 1 (%)")
    status_idx = header.index("Status")
    date_idx = header.index("实验日期")

    found = False
    for i in range(1, len(rows)):
        if rows[i][name_idx] == args.exp_name:
            rows[i][mean_idx] = mean_str
            rows[i][r1_idx] = result_strs[0]
            rows[i][r1_idx + 1] = result_strs[1]
            rows[i][r1_idx + 2] = result_strs[2]
            rows[i][r1_idx + 3] = result_strs[3]
            rows[i][r1_idx + 4] = result_strs[4]
            rows[i][status_idx] = "已完成"
            rows[i][date_idx] = args.date
            found = True
            break

    if not found:
        # 追加新行：列顺序与 RESULT.csv 一致（Base Model 在 Dataset 前）
        framework = meta["framework"] or "-"
        method = meta["method"] or "-"
        base_model = meta["base_model"] or "-"
        dataset = meta["dataset"] or "-"
        notes = meta["notes"] or ""
        new_row = [
            args.exp_name,
            framework,
            method,
            base_model,
            dataset,
            mean_str,
            result_strs[0],
            result_strs[1],
            result_strs[2],
            result_strs[3],
            result_strs[4],
            "已完成",
            notes,
            args.date,
        ]
        rows.append(new_row)
        print(f"已在 RESULT.csv 中追加新行: {args.exp_name}（请检查 Framework/Method/Dataset/Notes/Base Model 是否需手工补全）")

    save_result_csv(result_path, rows)
    print(f"已更新 {result_path}: {args.exp_name} Mean={mean}% Result1-5={result_strs}")


if __name__ == "__main__":
    main()
