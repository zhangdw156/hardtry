# verl11：4B-Thinking + EGPO + reward_fn_egpo（hardgen_1k_shuffle）

实验约定与脚本用法（bin 工具、用卡数等）见 [docs/exps-commons.md](../../docs/exps-commons.md)。

**26-03-02 实验**：在 **hardgen_1k_shuffle**（从 hardgen 14k 随机采样 1k）上，使用 **Qwen3-4B-Thinking** + **EGPO** + **reward_fn_egpo** 做强化学习。与 verl10（GRPO）、verl12（Instruct+GRPO）、full10/full11（SFT）共用同一数据，便于横向对比。

## 实验目的

- 验证在 hardgen_1k_shuffle 上 4B-Thinking + EGPO 的表现。
- 与 verl10（GRPO）公平对比：同数据、同奖励（严格二元 0/1）、同超参，仅 `adv_estimator` 为 egpo（并启用 egpo_lambda、egpo_alpha）。

## 数据

- **来源**：`data/hardgen/hardgen_openai_messages_fc.json`（hardgen 14k），经 `convert_messages_to_verl` **随机采样 1k**（`shuffle: true`）转为 parquet。
- **输出目录**：`data/hardgen_1k_shuffle/`（`train.parquet`、`test.parquet`），与 verl10/verl12/full10/full11 共用。
- **规模**：`max_samples=1000`，`shuffle=true`，`test_size=0.05`，`seed=42`。

## 算法与奖励

- **算法**：`adv_estimator: egpo`，`egpo_lambda: 0.4`，`egpo_alpha: 2.0`，`norm_adv_by_std_in_grpo: true`。EGPO 在 GRPO 的 advantage 上增加裁剪后的熵项，促进探索。
- **基座**：Qwen3-4B-Thinking-2507。
- **奖励**：`src/hardtry/rl/reward_fn_egpo.py` 的 `compute_score`（严格二元：格式正确且工具调用与 ground_truth 一致为 1.0，否则 0.0）。

## 配置与脚本

实验仅使用 **verl_meta_config.yaml** 与 **verl_train_config.yaml** 两个主配置（与 verl10/verl12 一致），EGPO 相关参数写在 `verl_train_config.yaml` 的 `algorithm` 与 `verl_meta_config.yaml` 的 `reward_fn_path` 中。

| 文件 / 脚本 | 说明 |
|-------------|------|
| `configs/verl_meta_config.yaml` | 实验元信息：数据根、reward 路径（reward_fn_egpo）、base 模型等 |
| `configs/verl_train_config.yaml` | 训练配置：EGPO 算法、数据、actor/rollout、trainer 等，引用 verl_meta_config |
| `configs/convert_messages_to_verl_config.yaml` | hardgen → hardgen_1k_shuffle parquet 的转换配置 |
| `configs/vllm_config.yaml` | 评估时 vLLM 服务配置 |
| `configs/eval_config5.yaml` | 评估配置 |
| `scripts/convert_messages_to_verl.sh` | 执行一次 convert，生成 hardgen_1k_shuffle |
| `scripts/train_local.sh` | 调用 verl.main_ppo，config-name=verl_train_config |
| `scripts/merge_verl_fsdp_local.sh` | 转调 commons/sbin/merge_verl_fsdp_auto.sh |
| `scripts/eval_local.sh` | 转调 commons/sbin/eval_local.sh |
| `run_train_only.sh` | 仅 train → merge → eval（不 convert），用于与 verl10/verl12 共数据对比 |

## 如何运行

### 方式一：与 verl10/verl12 使用相同数据（推荐）

1. **只执行一次 convert**（verl10、verl11 或 verl12 任选其一）：
   ```bash
   bash exps/verl11/scripts/convert_messages_to_verl.sh
   ```
2. **仅训练与评估**：
   ```bash
   bash exps/verl11/run_train_only.sh
   ```

### 方式二：单实验一条龙

```bash
bash exps/verl11/run_local.sh
```

会依次执行：convert → train (EGPO) → merge → eval。

## 输出位置

- **Checkpoint**：`/dfs/data/work/hardtry/checkpoints/verl11/`
- **Merge 后模型**：`/dfs/data/models/hardtry-4b-verl11`
- **评估结果**：`exps/verl11/eval5_results/`
