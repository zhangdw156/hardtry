# RL 奖励函数（VeRL 自定义 reward）

本目录提供三种 VeRL 自定义奖励入口，均实现 `compute_score(data_source, solution_str, ground_truth, extra_info=None)`，通过解析 `<tool_call>...</tool_call>` 与 ground_truth 比较。

| 文件 | 用途 | 适用实验 |
|------|------|----------|
| **reward_fn.py** | 格式 + 正确性分开计分：有 tool_call 给 0.1，与 gt 一致再给 1.0，总分 0 / 0.1 / 1.0 / 1.1 | 一般 GRPO |
| **reward_fn_grpo.py** | 严格 0/1，不要求 `<think>...</think>` 块，对整段 solution 做 tool_call 比较 | 不输出 think 的模型（如 Qwen3-4B-Instruct、verl9） |
| **reward_fn_egpo.py** | 严格 0/1，要求 `<think>...</think>` 块，仅对 </think> 后内容做 tool_call 比较 | EGPO、与 EGPO 公平对比的 GRPO（如 verl7/verl8） |

在 `verl_meta_config.yaml` 中通过 `reward_fn_path` 指定脚本路径，入口函数名为 `compute_score`。
