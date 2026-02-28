# verl8：EGPO 实验（hardgen 1k）

在 hardgen 1k 数据上使用 **EGPO**（Entropy-Guided Policy Optimization）做强化学习。与 verl7（GRPO）使用相同数据、相同奖励函数及相同共同参数，仅 advantage 估计算法为 EGPO。

## 实验目的

- 验证在 hardgen 1k 上 EGPO 的表现。
- 与 verl7（GRPO）公平对比：同数据、同奖励（严格二元 0/1）、同超参，仅 `adv_estimator` 为 egpo（并启用 egpo_lambda、egpo_alpha）。

## 数据

- **来源**：`data/hardgen/hardgen_openai_messages_fc.json`，经 `convert_messages_to_verl` 转为 verl 所需 parquet。
- **输出目录**：`data/hardgen_1k/`（`train.parquet`、`test.parquet`），与 verl7 共用。
- **规模**：`max_samples=1000`，`test_size=0.05`，`seed=42`（与 verl7 的 convert 配置一致）。

## 算法与奖励

- **算法**：`adv_estimator: egpo`，`egpo_lambda: 0.4`，`egpo_alpha: 2.0`，`norm_adv_by_std_in_grpo: true`。EGPO 在 GRPO 的 advantage 上增加裁剪后的熵项，促进探索（参见论文 *Reasoning through Exploration*, Hao et al. 2025）。
- **奖励**：与 verl7 相同，使用 `src/hardtry/rl/reward_fn_egpo.py` 的 `compute_score`（严格二元：格式正确且工具调用与 ground_truth 一致为 1.0，否则 0.0；与同目录 `reward_fn.py` 区分）。

## 配置与脚本

| 文件 / 脚本 | 说明 |
|-------------|------|
| `configs/verl_common_config_egpo.yaml` | EGPO 算法配置、数据路径、奖励函数等（主用） |
| `configs/egpo_config.yaml` | Hydra 入口，experiment_name=verl8，checkpoint 目录 |
| `configs/convert_messages_to_verl_config.yaml` | hardgen → hardgen_1k parquet 的转换配置 |
| `configs/vllm_config.yaml` | 评估时 vLLM 服务配置（model 指向 merge 后的权重；卡数由 set_exp_gpus.sh 管理） |
| `configs/eval_config5.yaml` | 评估配置（结果输出到本实验目录） |
| `scripts/convert_messages_to_verl.sh` | 执行一次 convert，生成 hardgen_1k |
| `scripts/train_local.sh` | 调用 verl.main_ppo，**config-name=egpo_config** |
| `scripts/merge_verl_fsdp_local.sh` | 将 checkpoint 合并为单模型，供 vLLM 加载 |
| `scripts/eval_local.sh` | 启动 vLLM 并跑评估 |

## 如何运行

### 方式一：保证与 verl7 使用完全相同的训练集与测试集（推荐）

1. **只执行一次 convert**（verl7 或 verl8 任选其一执行即可）：
   ```bash
   bash exps/verl7/scripts/convert_messages_to_verl.sh
   ```
2. **仅训练与评估**（不再次 convert）：
   ```bash
   bash exps/verl8/run_train_only.sh
   ```
   对 verl7 执行：`bash exps/verl7/run_train_only.sh`。  
   这样 verl7 与 verl8 读的是同一份 `data/hardgen_1k/train.parquet` 与 `test.parquet`，可保证数据完全一致。

### 方式二：单实验一条龙（convert + train + merge + eval）

若只跑 verl8、不要求与 verl7 严格共数据，可直接：

```bash
bash exps/verl8/run_local.sh
```

会依次执行：convert → train (EGPO) → merge → eval。

## 输出位置

- **Checkpoint**：`/dfs/data/work/hardtry/checkpoints/verl8/`（由 `egpo_config.yaml` 的 `default_local_dir` 决定）。
- **Merge 后模型**：`/dfs/data/models/hardtry-4b-verl8`（由 `scripts/merge_verl_fsdp_local.sh` 的 `TARGET_DIR` 决定）。
- **评估结果**：`exps/verl8/eval5_results/`（由 `configs/eval_config5.yaml` 的 `summary_output_dir` 决定）。

## 依赖与环境

- 需已安装 **verl**（含 **EGPO** 实现），并配置好 Ray、vLLM 等。
- 训练使用 `configs` 中指定的 venv/路径（如 `VERL_VENV` 在 `scripts/train_local.sh` 中）；评估使用 `eval_config5.yaml` 中的 `venv_activate_path`。
