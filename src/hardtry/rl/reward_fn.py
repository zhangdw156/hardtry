"""
通用 tool_call 奖励（格式 + 正确性分开计分）：有 <tool_call> 给 0.1 格式分，
解析后与 ground_truth 一致再给 1.0 正确性分，总分 0 / 0.1 / 1.0 / 1.1。
适用于一般 GRPO 等；需严格 0/1 时用 reward_fn_egpo.py，不要求 <think> 块时用 reward_fn_grpo.py。
"""
import json
import re
from collections import Counter


def _convert_to_hashable(data):
    """将 dict/list 转为可哈希类型，用于忽略顺序的比较。"""
    if isinstance(data, dict):
        return frozenset(
            (key, _convert_to_hashable(value)) for key, value in data.items()
        )
    if isinstance(data, list):
        return frozenset(_convert_to_hashable(item) for item in data)
    return data


def _compare_parsed_content(parsed1, parsed2):
    """比较两个工具调用列表，忽略顺序。"""
    counter1 = Counter([_convert_to_hashable(item) for item in parsed1])
    counter2 = Counter([_convert_to_hashable(item) for item in parsed2])
    return counter1 == counter2


def _extract_tool_calls(input_string):
    """从文本中提取 <tool_call> 标签内的 JSON 内容。"""
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, input_string, re.DOTALL)
    result = []
    for match in matches:
        try:
            result.append(json.loads(match))
        except Exception:
            result.append(match)
    return result


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VeRL 自定义奖励入口：格式分（0.1）+ 正确性分（0 或 1.0），总分 0 / 0.1 / 1.0 / 1.1。
    Args:
        data_source: 数据来源标识（未用）
        solution_str: 模型生成的完整字符串
        ground_truth: 标准答案（含 <tool_call>）
        extra_info: 额外信息（未用）
    """
    format_reward = 0.0
    correctness_reward = 0.0
    has_tool = "<tool_call>" in solution_str and "</tool_call>" in solution_str
    if has_tool:
        format_reward = 0.1
    try:
        gt_tools = _extract_tool_calls(ground_truth)
        pd_tools = _extract_tool_calls(solution_str)
        if pd_tools and _compare_parsed_content(gt_tools, pd_tools):
            correctness_reward = 1.0
    except Exception:
        pass
    return format_reward + correctness_reward
