"""
verl7_demo 专用奖励函数：与 src/hardtry/rl/reward_fn_egpo.py 逻辑一致，
但加入多处 print，用于验证：
  1) 模块被正确加载
  2) compute_score 被 VeRL 正确调用
  3) 输入/中间/输出可追溯，确认奖励函数有用

EGPO 严格二元：<think>...</think>... 格式 + </think> 后 tool_call 与 ground_truth 一致 → 1.0，否则 0.0。
"""
import json
import re
from collections import Counter

# ----- 验证 1：模块被正确加载 -----
print("[verl7_demo reward_fn] 模块已加载 (reward_fn.py)")

# 单条单轮下只会有 1 次调用，始终完整打印即可，不刷屏
_COMPUTE_SCORE_CALL_COUNT = 0


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
    """
    校验格式为 <think>...</think>...，并返回 </think> 之后的内容。
    若不存在完整的 <think>...</think> 或 </think> 后无内容，返回 None。
    """
    if not solution_str or "<think>" not in solution_str or "</think>" not in solution_str:
        return None
    match = re.search(r"<think>.*?</think>\s*(.*)", solution_str, re.DOTALL)
    if not match:
        return None
    after = match.group(1).strip()
    return after if after else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    EGPO 严格二元奖励入口。单条单轮下只调 1 次，此处完整打印以验证「被正确加载并应用」。
    """
    global _COMPUTE_SCORE_CALL_COUNT
    _COMPUTE_SCORE_CALL_COUNT += 1
    call_num = _COMPUTE_SCORE_CALL_COUNT

    print(f"\n[verl7_demo reward_fn] ----- compute_score 第 {call_num} 次调用 -----")
    print(f"  data_source: {data_source}")
    print(f"  solution_str (前 500 字符): {(solution_str or '')[:500]!r}")
    print(f"  ground_truth (前 300 字符): {(str(ground_truth).strip() or '')[:300]!r}")
    print(f"  extra_info: {extra_info}")

    if not (solution_str and str(ground_truth).strip()):
        print("  -> 输入为空，返回 0.0")
        return 0.0

    content_after_think = _extract_after_think(solution_str)
    print(f"  content_after_think: {content_after_think[:200] if content_after_think else None!r}")

    if content_after_think is None:
        print("  -> 无 <think>...</think>... 格式，返回 0.0")
        return 0.0

    has_tool = "<tool_call>" in content_after_think and "</tool_call>" in content_after_think
    print(f"  has_tool: {has_tool}")

    if not has_tool:
        print("  -> 无 tool_call，返回 0.0")
        return 0.0

    try:
        gt_tools = extract_tool_calls(ground_truth)
        pd_tools = extract_tool_calls(content_after_think)
        print(f"  gt_tools: {gt_tools}")
        print(f"  pd_tools: {pd_tools}")
        if not pd_tools:
            print("  -> pd_tools 为空，返回 0.0")
            return 0.0
        match = compare_parsed_content(gt_tools, pd_tools)
        print(f"  compare_parsed_content: {match}")
        if match:
            print("  -> 一致，返回 1.0")
            return 1.0
    except Exception as e:
        print(f"  -> 异常: {e}，返回 0.0")
        return 0.0
    print("  -> 默认返回 0.0")
    return 0.0
