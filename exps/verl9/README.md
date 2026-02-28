# verl9：GRPO + Qwen3-4B-Instruct（hardgen 1k）

实验约定与脚本用法（bin 工具、用卡数等）见 [docs/exps-commons.md](../../docs/exps-commons.md)。

在 hardgen 1k 数据上使用 **GRPO** 训练 **Qwen3-4B-Instruct-2507**。与 verl7/verl8 使用相同数据与相同训练超参，但奖励使用 `reward_fn_grpo.py`（二元 0/1，不要求 `<think>` 格式），适用于不输出思考块的 Instruct 模型。

## 实验目的

- 在相同数据与超参下，对比 GRPO 在 **Instruct** 基座（verl9）与 **Thinking** 基座（verl7）上的表现。
- 与 verl7（GRPO + Thinking）、verl8（EGPO + Thinking）共用 hardgen_1k 数据，便于横向对比。

## 数据

- **来源**：与 verl7/verl8 相同，`data/hardgen/hardgen_openai_messages_fc.json`，经 convert 转为 `data/hardgen_1k/`。
- **输出目录**：`data/hardgen_1k/`（`train.parquet`、`test.parquet`），与 verl7/verl8 共用。
- **规模**：`max_samples=1000`，`test_size=0.05`，`seed=42`。

## 算法与奖励

- **算法**：`adv_estimator: grpo`，`norm_adv_by_std_in_grpo: true`（与 verl7 一致）。
- **基座模型**：Qwen3-4B-Instruct-2507（非 Thinking，不要求输出 `<think>...</think>`）。
- **奖励**：`src/hardtry/rl/reward_fn_grpo.py` 的 `compute_score`（二元 0/1：整段 solution 中 tool_call 与 ground_truth 一致为 1.0，否则 0.0；不检查 think 块）。

## 配置与脚本

| 文件 / 脚本 | 说明 |
|-------------|------|
| `configs/verl_config.yaml` | 实验元信息：数据根 hardgen_1k、reward_fn_grpo、base 模型 Instruct |
| `configs/verl_common_config.yaml` | 算法、数据路径、actor/rollout、trainer、奖励引用等 |
| `configs/convert_messages_to_verl_config.yaml` | hardgen → hardgen_1k 转换配置（与 verl7 一致） |
| `configs/vllm_config.yaml` | 评估时 vLLM 配置（model 指向 merge 后权重） |
| `configs/eval_config5.yaml` | 评估配置 |
| `scripts/convert_messages_to_verl.sh` | 执行一次 convert（与 verl7/verl8 共用输出目录即可） |
| `scripts/train_local.sh` | GRPO 训练，config-name=verl_common_config |
| `scripts/merge_verl_fsdp_local.sh` | 合并 checkpoint 为单模型 |
| `scripts/eval_local.sh` | 转调 commons/sbin/eval_local.sh |
| `run_train_only.sh` | 仅 train → merge → eval（不 convert），用于与 verl7/verl8 共数据对比 |

## 如何运行

### 方式一：与 verl7/verl8 使用相同数据（推荐）

1. **只执行一次 convert**（verl7、verl8 或 verl9 任选其一）：
   ```bash
   bash exps/verl7/scripts/convert_messages_to_verl.sh
   ```
2. **仅训练与评估**（不再次 convert）：
   ```bash
   bash exps/verl9/run_train_only.sh
   ```

### 方式二：单实验一条龙

```bash
bash exps/verl9/run_local.sh
```

会依次执行：convert → train (GRPO) → merge → eval。

## 输出位置

- **Checkpoint**：`/dfs/data/work/hardtry/checkpoints/verl9/`（由 `verl_config.checkpoint_dir` 决定）。
- **Merge 后模型**：`/dfs/data/models/hardtry-4b-verl9`（由 `scripts/merge_verl_fsdp_local.sh` 的 `TARGET_DIR` 决定）。
- **评估结果**：`exps/verl9/eval5_results/`（由 `configs/eval_config5.yaml` 的 `summary_output_dir` 决定）。

## 依赖与环境

- 需已安装 **verl**（含 GRPO），并配置好 Ray、vLLM 等。
- 训练使用 `verl_config.venv_verl`；评估使用 `eval_config5.yaml` 中的 `venv_activate_path`。
