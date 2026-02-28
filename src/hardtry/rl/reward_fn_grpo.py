"""
GRPO 用二元 reward（不要求 <think> 格式）：仅当 solution 中存在 <tool_call> 且解析后与 ground_truth 一致时 1.0，否则 0.0。
用于 Qwen3-4B-Instruct 等不输出 <think>...</think> 的模型（如 verl9），与 reward_fn_egpo.py（要求 think 块）区分。
"""

"""
奖励计算通用工具：工具调用解析与比较。
供 reward_fn.py、reward_fn_egpo.py 等复用。
"""
import json
import re
from collections import Counter


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
