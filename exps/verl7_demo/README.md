# verl7_demo

用于**验证 VeRL 可跑通**且**奖励函数被正确加载并应用**的轻量实验。基于 verl7（GRPO + hardgen_1k），仅将奖励函数替换为本目录下的 `reward_fn.py`，其中带有详细打印。

## 验证什么

1. **VeRL 可行**：能正常完成数据加载 → rollout → reward → actor/ref 更新的一轮训练。
2. **奖励函数有用**：通过日志中的 `[verl7_demo reward_fn]` 输出依次确认：
   - **模块已加载**：启动后出现 `[verl7_demo reward_fn] 模块已加载 (reward_fn.py)`，说明自定义 reward 被正确加载。
   - **compute_score 被调用**：出现多段 `compute_score 第 k 次调用` 或 `compute_score #k score=...`，说明 VeRL 在算 reward 时调用了该函数。
   - **输入与逻辑可追溯**：前几次调用会完整打印 `data_source`、`solution_str`、`ground_truth`、`extra_info` 以及中间步骤（如 `content_after_think`、`has_tool`、`gt_tools`/`pd_tools`、最终 0.0/1.0），便于核对奖励逻辑是否正确。

## 与 verl7 的差异

| 项目       | verl7                    | verl7_demo                          |
|------------|--------------------------|-------------------------------------|
| 算法/数据  | 与 verl7 相同（grpo + hardgen_1k） | 继承 verl7 的 `verl_common_config` |
| 奖励函数   | `src/hardtry/rl/reward_fn_egpo.py` | `exps/verl7_demo/reward_fn.py`（逻辑一致，加打印） |
| 实验名/ckpt | verl7                   | verl7_demo                          |

## 如何跑

```bash
cd /dfs/data/work/hardtry/exps/verl7_demo
bash scripts/train_local.sh
```

日志会写入 `logs/grpo.log`，同时 tee 到终端。在日志中搜索 `[verl7_demo reward_fn]` 即可查看奖励函数的加载与每次调用详情。

## 文件说明

- `reward_fn.py`：与 `reward_fn_egpo.py` 相同的严格二元逻辑（<think>...</think> + tool_call 一致 → 1.0），增加模块加载与 `compute_score` 的打印。
- `configs/grpo_config.yaml`：继承 verl7 的 common 配置，仅覆盖实验名、checkpoint 目录和 reward 路径。
- `scripts/train_local.sh`：仅执行 GRPO 训练，不包含 convert/merge/eval。
