#!/usr/bin/env python

import ast
import json
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import HfArgumentParser
import re


@dataclass
class DataArguments:
    input: str = field(metadata={"help": "Input raw json path"})
    output: str = field(metadata={"help": "Output messages jsonl path"})
    max_samples: int = field(
        default=-1,
        metadata={"help": "How many samples to convert. If < 0, convert all samples."},
    )


# =================配置区域=================
NEW_INSTRUCTION = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the
question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the
function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of <tool_call>...</tool_call>
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn.
Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once
you have no more functions to call, the system will consider the current turn complete and proceed to the next turn
or task."""

# =================AST 解析工具函数=================


def _get_node_name(node):
    """递归提取复杂的函数名。支持: print, os.path.join, tools['search'], get_tool()"""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_get_node_name(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        value = _get_node_name(node.value)
        if isinstance(node.slice, ast.Constant):
            slice_val = repr(node.slice.value)
        else:
            slice_val = ast.unparse(node.slice)
        return f"{value}[{slice_val}]"
    else:
        return ast.unparse(node)


def _get_arg_value(node):
    """解析参数值：优先转为原生对象，失败则回退为源码字符串。"""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return ast.unparse(node)


def convert_python_to_xml_structure(input_str: str) -> list:
    """将 Python 列表格式的调用转换为结构化列表。"""
    result_parts = []
    try:
        tree = ast.parse(input_str.strip(), mode="eval")
        if not isinstance(tree.body, ast.List):
            return []
        for node in tree.body.elts:
            if isinstance(node, ast.Call):
                func_name = _get_node_name(node.func)
                args_dict = {}
                # 处理位置参数
                for i, arg in enumerate(node.args):
                    args_dict[f"arg_{i}"] = _get_arg_value(arg)
                # 处理关键字参数
                for keyword in node.keywords:
                    args_dict[keyword.arg] = _get_arg_value(keyword.value)

                result_parts.append(
                    {
                        "name": func_name,
                        "arguments": json.dumps(args_dict, ensure_ascii=False),
                    }
                )
    except Exception:
        return []
    return result_parts


# =================角色转换核心逻辑=================


def convert_system_content(example):
    tools_text = ""
    if "tools" in example and example["tools"]:
        try:
            tools_obj = json.loads(example["tools"])
            if isinstance(tools_obj, dict):
                tools_obj = [tools_obj]

            tools_header = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            tools_json_body = "".join(
                ["\n" + json.dumps(t, ensure_ascii=False) for t in tools_obj]
            )
            tools_footer = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
            tools_text = tools_header + tools_json_body + tools_footer
        except Exception:
            tools_text = ""
    return {"role": "system", "content": NEW_INSTRUCTION + tools_text}


def convert_assistant_content(example):
    content = example["content"]
    # 移除 think 标签
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()

    # 如果是 Python 列表格式，转为 XML 格式
    if content.startswith("[") and content.endswith("]"):
        tool_calls = convert_python_to_xml_structure(content)
        if not tool_calls:
            return {"role": "error", "content": ""}

        xml_content = ""
        for tc in tool_calls:
            xml_content += f'<tool_call>\n{{"name": "{tc["name"]}", "arguments": {tc["arguments"]}}}\n</tool_call>\n'
        content = xml_content.strip()

    return {"role": "assistant", "content": content}


def convert_tool_content(example):
    """将工具返回结果包装为 user 角色（模拟系统反馈）。"""
    try:
        tool_calls = ast.literal_eval(example["content"])
        content = ""
        for tool_call in tool_calls:
            for value in tool_call.values():
                # TODO: 修改了对于函数调用错误的处理
                # 需要当前几组实验跑完之后重新构造数据再进行实验
                # 不过就目前结果而看，应该影响不大，所以暂时先不调用
                if isinstance(value,str):
                    pattern = r"^Function call .* failed\..*?Error:\s+(.*?)\s+Stack trace:"
                    match = re.search(pattern, value, re.DOTALL)
                    if match:
                        value = {"error": match.group(1)}
                content += f"<tool_response>\n{value}\n</tool_response>\n"
        return {"role": "user", "content": content.strip()}
    except Exception:
        return {"role": "error", "content": ""}


# =================主转换流程=================


def convert_messages(example):
    messages = []
    # 1. 系统消息
    if example.get("0"):
        messages.append(convert_system_content(example.get("0")))

    # 2. 轮次消息
    turn_keys = sorted([k for k in example.keys() if k.isdigit() and k != "0"], key=int)
    for key in turn_keys:
        turn_data = example[key]
        if not turn_data:
            continue

        role = turn_data.get("role")
        if role == "user":
            msg = {"role": "user", "content": turn_data["content"]}
        elif role == "assistant":
            msg = convert_assistant_content(turn_data)
        elif role == "tool":
            msg = convert_tool_content(turn_data)

        if msg.get("role") == "error":
            return {"messages": []}
        messages.append(msg)

    return {"messages": messages}


def truncate_at_last_tool_call(example):
    """
    保留对话直到最后一个包含 <tool_call> 的 assistant 消息。
    """
    messages = example["messages"]
    if not messages:
        return {"messages": []}

    last_valid_index = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
            last_valid_index = i
            break

    if last_valid_index != -1:
        return {"messages": messages[: last_valid_index + 1]}
    return {"messages": []}


# =================执行脚本=================


def main():
    # 初始化解析器
    parser = HfArgumentParser(DataArguments)

    # 支持从 .yaml 文件读取，也可以从命令行覆盖参数
    # 运行命令示例: python script.py config.yaml 或 python script.py --input a.json --output b.jsonl
    import sys

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # 如果只传了一个 yaml 文件路径
        data_args = parser.parse_yaml_file(yaml_file=sys.argv[1])[0]
    else:
        # 否则按常规命令行参数解析
        data_args = parser.parse_args_into_dataclasses()[0]

    print(f"Loading dataset from {data_args.input}...")
    ds = load_dataset("json", data_files=data_args.input, split="train")

    # 0. 采样
    if data_args.max_samples >= 0:
        # 取设定值和数据集实际长度的最小值，防止溢出
        num_to_select = min(data_args.max_samples, len(ds))
        print(f"Selecting first {num_to_select} samples...")
        ds = ds.select(range(num_to_select))

    # 1. 转换为基础 messages 格式
    print("Converting to messages format...")
    ds = ds.map(convert_messages, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["messages"] is not None and len(x["messages"]) > 0)

    # 2. 截断对话（保证以 tool_call 结尾）
    print("Truncating conversations at last tool_call...")
    ds = ds.map(truncate_at_last_tool_call)
    ds = ds.filter(lambda x: len(x["messages"]) > 0)

    # 3. 保存为 JSON
    print(f"Saving {len(ds)} samples to {data_args.output}...")
    with open(data_args.output, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Done!")


if __name__ == "__main__":
    main()
