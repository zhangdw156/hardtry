# full8：全量 SFT（Thinking 基座，hardgen_1k）

实验约定与脚本用法见 [docs/exps-commons.md](../../docs/exps-commons.md)。

在 **hardgen_1k** 数据上对 **Qwen3-4B-Thinking-2507** 做全量 SFT，训练与评估数据与 RL 实验（verl7/verl8/verl9）一致。

## 数据与 verl7/verl8 一致（如何体现）

| 环节 | verl7/verl8 | full8/full9 |
|------|--------------|-------------|
| **原始数据** | `convert_messages_to_verl` 产出 `data/hardgen_1k/train.parquet`、`test.parquet` | 无单独 convert；直接使用上述 parquet 的**导出结果** |
| **数据文件** | `train_files` / `val_files` 指向上述两个 parquet | `dataset` → `train.jsonl`，`val_dataset` → `test.jsonl`（与 parquet 一一对应） |
| **同数据保证** | — | `train.jsonl` / `test.jsonl` 由 `parquet_to_openai_messages --output_dir data/hardgen_1k` 生成：**train.parquet → train.jsonl**，**test.parquet → test.jsonl**，与 RL 同一份样本、同一划分。 |

因此：先有 verl7/verl8 使用的 parquet，再对该目录跑一次导出得到 `train.jsonl`、`test.jsonl`，full8/full9 中**分开指定**为 dataset / val_dataset，即与 verl7/verl8 同数据。

## 数据与划分

- **训练**：`data/hardgen_1k/train.jsonl`（由 train.parquet 导出）。
- **验证**：`data/hardgen_1k/test.jsonl`（由 test.parquet 导出），与 RL 的 test 集一致。

## 模型与模板

- **基座**：Qwen3-4B-Thinking-2507  
- **template**：`qwen3`（带思考格式）

## 运行前准备

**当前仅有 `data/hardgen_1k/train.parquet`、`test.parquet` 时，`train.jsonl` / `test.jsonl` 尚不存在，必须先执行导出。**

1. **生成 SFT 用数据**（在存在 parquet 的机器上，如 Linux 训练机）执行一次：
   ```bash
   bash exps/full8/scripts/export_hardgen_1k_openai.sh
   ```
   或等价命令：
   ```bash
   uv run python -m hardtry.utils.parquet_to_openai_messages \
     --input_dir /dfs/data/work/hardtry/data/hardgen_1k \
     --output_dir /dfs/data/work/hardtry/data/hardgen_1k
   ```
   会在 `data/hardgen_1k/` 下生成 **train.jsonl**（对应 train.parquet）、**test.jsonl**（对应 test.parquet）。

2. **运行实验**：
   ```bash
   bash exps/full8/run_local.sh
   ```
   依次执行：train (SFT) → merge → eval。

## 输出

- **Checkpoint**：`/dfs/data/work/hardtry/checkpoints/full8/`
- **Merge 后模型**：`/dfs/data/models/hardtry-4b-full8`
- **评估结果**：`exps/full8/eval5_results/`
