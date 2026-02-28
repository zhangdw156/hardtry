"""
RL 奖励模块：通用工具与多种奖励实现。

- reward_utils：可复用工具（解析 tool_call、比较结果）
- reward_fn：通用分段计分（格式 + 正确性，如 0/0.1/1.0/1.1）
- reward_fn_egpo：EGPO 用严格二元 0/1 奖励
"""
from hardtry.rl.reward_utils import (
    compare_parsed_content,
    convert_to_hashable,
    extract_tool_calls,
)
from hardtry.rl.reward_fn import compute_score as compute_score_grpo
from hardtry.rl.reward_fn_egpo import compute_score as compute_score_egpo

__all__ = [
    "compare_parsed_content",
    "convert_to_hashable",
    "extract_tool_calls",
    "compute_score_grpo",
    "compute_score_egpo",
]
