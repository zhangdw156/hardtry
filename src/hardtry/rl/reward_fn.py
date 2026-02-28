"""
通用 tool_call 奖励：格式与正确性分开计分（如 0 / 0.1 / 1.0 / 1.1）。
EGPO 及需要严格二元 0/1 时请使用同目录 reward_fn_egpo.py。
"""
from hardtry.rl.reward_utils import compare_parsed_content, extract_tool_calls


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VeRL 自定义奖励函数入口
    Args:
        solution_str: 模型生成的完整字符串
        ground_truth: 标准答案
    """
    format_reward = 0.0
    correctness_reward = 0.0

    predict_str = solution_str
    
    has_tool = "<tool_call>" in predict_str and "</tool_call>" in predict_str

    if has_tool:
        format_reward = 0.1

    try:
        gt_tools = extract_tool_calls(ground_truth)
        pd_tools = extract_tool_calls(predict_str)

        if not pd_tools:
            correctness_reward = 0.0
        elif compare_parsed_content(gt_tools, pd_tools):
            correctness_reward = 1.0
    except Exception:
        correctness_reward = 0.0

    total_score = format_reward + correctness_reward

    return total_score
