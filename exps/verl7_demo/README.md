# verl7_demo

**单条数据、单轮**：用 1 条样本跑 1 个 step，验证 VeRL 可跑通且奖励函数被正确加载并应用，日志不刷屏。

基于 verl7（GRPO + hardgen_1k），覆盖为：1 样本 / 1 batch / 1 GPU / 1 epoch，奖励函数用本目录 `reward_fn.py`（带一次完整打印）。

## 验证什么

1. **VeRL 可行**：能正常完成数据加载 → rollout → reward → actor/ref 更新的一轮训练。
2. **奖励函数有用**：日志里 `[verl7_demo reward_fn]` 只会出现**一次**完整调用：
   - **模块已加载**：`[verl7_demo reward_fn] 模块已加载 (reward_fn.py)`
   - **compute_score 被调用**：`compute_score 第 1 次调用` 及完整输入、中间步骤、返回 0.0/1.0

## 与 verl7 的差异

| 项目         | verl7                         | verl7_demo |
|--------------|-------------------------------|------------|
| 数据/轮数    | 64 batch，多 step             | 1 样本、1 batch、1 epoch（1 step） |
| GPU          | 2                             | 1 |
| 奖励函数     | `src/hardtry/rl/reward_fn_egpo.py` | `exps/verl7_demo/reward_fn.py`（逻辑一致，单次完整打印） |
| 实验名/ckpt  | verl7                         | verl7_demo |

## 如何跑

```bash
cd /dfs/data/work/hardtry/exps/verl7_demo
bash scripts/train_local.sh
```

日志写入 `logs/grpo.log` 并 tee 到终端。搜 `[verl7_demo reward_fn]` 即可看到一次完整奖励调用，不刷屏。

## 文件说明

- `reward_fn.py`：与 `reward_fn_egpo.py` 相同的严格二元逻辑，增加模块加载与单次 `compute_score` 的完整打印。
- `configs/grpo_config.yaml`：继承 verl7 common，覆盖为单条单轮（train_max_samples=1、train_batch_size=1、n_gpus_per_node=1、rollout.n=1、total_epochs=1 等）。
- `scripts/train_local.sh`：仅执行 GRPO 训练。
