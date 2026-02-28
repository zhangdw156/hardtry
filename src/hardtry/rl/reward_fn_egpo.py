"""
EGPO 用严格二元 reward（Hao et al. 2025）：
仅当「格式正确（含 <think>...</think>... 思考块 + 后续内容）」且「思考块后的内容中工具调用与 ground_truth 一致」时返回 1.0，否则 0.0。

与同目录下 reward_fn.py 的区分：
- reward_fn.py：格式与正确性分开计分，如 0 / 0.1 / 1.0 / 1.1，用于一般 GRPO 等。
- reward_fn_egpo.py：严格 0/1，用于 EGPO 及与 EGPO 公平对比的 GRPO（如 verl7/verl8）。
"""
import json
import re
from collections import Counter


def _convert_to_hashable(data):
    if isinstance(data, dict):
        return frozenset((key, _convert_to_hashable(value)) for key, value in data.items())
    if isinstance(data, list):
        return frozenset(_convert_to_hashable(item) for item in data)
    return data


def compare_parsed_content(parsed1, parsed2):
    """比较两个工具调用列表，忽略顺序。"""
    counter1 = Counter([_convert_to_hashable(item) for item in parsed1])
    counter2 = Counter([_convert_to_hashable(item) for item in parsed2])
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
    return solution_str.rsplit(delimiter,1)[-1]


# 调试：前若干次调用打印输入，便于排查 verl7 critic/score/mean 恒为 0
_DEBUG_CALL_COUNT = 0
_DEBUG_MAX_PRINT = 5


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    EGPO 严格二元奖励：先验证 <think>...</think>... 格式，再对 </think> 后的内容做 AST（tool_call）校验；
    格式符合且 AST 全对才 1.0，否则 0.0。
    与 verl 自定义奖励入口签名一致。
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
