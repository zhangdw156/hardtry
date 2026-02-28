"""
GRPO 用二元 reward（不要求 <think> 格式）：仅当 solution 中存在 <tool_call> 且解析后与 ground_truth 一致时 1.0，否则 0.0。
用于 Qwen3-4B-Instruct 等不输出 <think>...</think> 的模型（如 verl9），与 reward_fn_egpo.py（要求 think 块）区分。
"""
from .reward_utils import compare_parsed_content, extract_tool_calls


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    GRPO 二元奖励：不检查 <think>；对整段 solution_str 做 tool_call 解析并与 ground_truth 比较；
    有 tool_call 且一致则 1.0，否则 0.0。与 VeRL 自定义奖励入口签名一致。
    """
    if not solution_str:
        return 0.0

    has_tool = "<tool_call>" in solution_str and "</tool_call>" in solution_str
    if not has_tool:
        return 0.0

    try:
        gt_tools = extract_tool_calls(ground_truth)
        pd_tools = extract_tool_calls(solution_str)
        if not pd_tools:
            return 0.0
        if compare_parsed_content(gt_tools, pd_tools):
            return 1.0
    except Exception:
        pass
    return 0.0
