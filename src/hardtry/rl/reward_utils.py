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
