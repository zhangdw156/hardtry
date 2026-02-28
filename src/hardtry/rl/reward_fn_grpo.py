"""
GRPO 用二元 reward（不要求 <think> 格式）：仅当 solution 中存在 <tool_call> 且解析后与 ground_truth 一致时 1.0，否则 0.0。
适用于不输出 <think>...</think> 的模型（如 Qwen3-4B-Instruct、verl9）。需 <think> 块时用 reward_fn_egpo.py；格式+正确性分开计分用 reward_fn.py。
"""
import json
import re
from collections import Counter

# 工具调用解析与比较（本模块自包含，与 reward_fn_egpo 等不共享以兼容 VeRL 加载方式）


def convert_to_hashable(data):
    """将 dict/list 转为可哈希类型，用于忽略顺序的比较。"""
    if isinstance(data, dict):
        return frozenset(
            (key, convert_to_hashable(value)) for key, value in data.items()
        )
    if isinstance(data, list):
        return frozenset(convert_to_hashable(item) for item in data)
    return data


def compare_parsed_content(parsed1, parsed2):
    """比较两个工具调用列表，忽略顺序。"""
    counter1 = Counter([convert_to_hashable(item) for item in parsed1])
    counter2 = Counter([convert_to_hashable(item) for item in parsed2])
    return counter1 == counter2


def extract_tool_calls(input_string):
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
    VeRL 自定义奖励入口。不检查 <think>，对整段 solution 做 <tool_call> 解析；
    有 tool_call 且与 ground_truth 一致返回 1.0，否则 0.0。
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
