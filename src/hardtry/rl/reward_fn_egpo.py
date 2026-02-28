"""
EGPO 用严格二元 reward（Hao et al. 2025）：仅当「含 <think>...</think> 思考块且格式正确」且「</think> 后内容中 tool_call 与 ground_truth 一致」时 1.0，否则 0.0。
用于 EGPO 及与 EGPO 公平对比的 GRPO（如 verl7/verl8）。不要求 think 块用 reward_fn_grpo.py；格式+正确性分开计分用 reward_fn.py。
"""
import json
import re
from collections import Counter

# 工具调用解析与比较（本模块自包含，与 reward_fn_grpo 等不共享以兼容 VeRL 加载方式）


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


def _extract_after_think(solution_str: str) -> str | None:
    delimiter = "</think>\n\n"
    if delimiter not in solution_str:
        return None
    return solution_str.rsplit(delimiter, 1)[-1]


# 调试：前若干次调用打印输入，便于排查 verl7 critic/score/mean 恒为 0
_DEBUG_CALL_COUNT = 0
_DEBUG_MAX_PRINT = 5


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VeRL 自定义奖励入口。先验证 <think>...</think> 格式，再对 </think> 后内容做 <tool_call> 解析与比较；
    格式正确且与 ground_truth 一致返回 1.0，否则 0.0。
    """
    global _DEBUG_CALL_COUNT
    _DEBUG_CALL_COUNT += 1
    if _DEBUG_CALL_COUNT <= _DEBUG_MAX_PRINT:
        print(
            f"[reward_fn_egpo] call #{_DEBUG_CALL_COUNT}\n"
            f"solution_preview={solution_str!r}\n"
            f"gt_preview={ground_truth!r}"
        )

    if not solution_str:
        return 0.0

    content_after_think = _extract_after_think(solution_str)
    if content_after_think is None:
        return 0.0

    has_tool = "<tool_call>" in content_after_think and "</tool_call>" in content_after_think
    if not has_tool:
        return 0.0

    try:
        gt_tools = extract_tool_calls(ground_truth)
        pd_tools = extract_tool_calls(content_after_think)
        if not pd_tools:
            return 0.0
        if compare_parsed_content(gt_tools, pd_tools):
            if _DEBUG_CALL_COUNT <= _DEBUG_MAX_PRINT:
                print("[reward_fn_egpo] -> 1.0 (match)")
            return 1.0
    except Exception as e:
        if _DEBUG_CALL_COUNT <= _DEBUG_MAX_PRINT:
            print(f"[reward_fn_egpo] exception: {e}")
    return 0.0
