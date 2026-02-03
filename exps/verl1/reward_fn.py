import json
import re
from collections import Counter


def compare_parsed_content(parsed1, parsed2):
    """比较两个工具调用列表，忽略顺序。"""

    def convert_to_hashable(data):
        if isinstance(data, dict):
            return frozenset(
                (key, convert_to_hashable(value)) for key, value in data.items()
            )
        elif isinstance(data, list):
            return frozenset(convert_to_hashable(item) for item in data)
        else:
            return data

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
            # 如果 JSON 格式非法，放入原字符串后续匹配会失败
            result.append(match)
    return result


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VeRL 自定义奖励函数入口
    Args:
        solution_str: 模型生成的完整字符串
        ground_truth: 标准答案
    """
    predict_str = (
        solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    )

    format_reward = 0.0
    correctness_reward = 0.0

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
