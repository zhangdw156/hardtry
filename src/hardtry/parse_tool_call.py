import ast
import json

def parse_tool_call(call_str):
    """
    将 Python 函数调用格式的字符串转换为 OpenAI Function Call 格式。
    
    Args:
        call_str (str): 例如 'pwd()' 或 'find(path=".")'
        
    Returns:
        dict: {"name": "func_name", "arguments": {key: value}}
    """
    try:
        # 1. 使用 ast.parse 解析字符串为语法树
        # mode='eval' 表示我们需要解析的是一个表达式
        tree = ast.parse(call_str.strip(), mode='eval')
        
        # 2. 确保解析结果是一个函数调用 (Call节点)
        if not isinstance(tree.body, ast.Call):
            return {"error": "输入的字符串不是有效的函数调用格式"}
            
        call_node = tree.body
        
        # 3. 提取函数名称
        # 处理简单的函数名，如 'find'
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
        # 处理带模块前缀的函数名，如 'os.path.join' (虽然Tool调用通常不带前缀)
        elif isinstance(call_node.func, ast.Attribute):
            func_name = ast.unparse(call_node.func)
        else:
            func_name = "unknown_function"

        # 4. 提取参数 (Keywords)
        arguments = {}
        
        # 遍历关键字参数 (key=value)
        for keyword in call_node.keywords:
            arg_name = keyword.arg
            # ast.literal_eval 可以安全地评估节点的值（处理字符串、数字、布尔值、列表等）
            # 注意：如果参数是非常复杂的表达式，这里可能需要更复杂的处理，
            # 但对于 Tool Call，参数通常是字面量 (Literals)。
            arg_value = ast.literal_eval(keyword.value)
            arguments[arg_name] = arg_value
            
        # *注意：OpenAI 格式通常不支持位置参数（如 find(".", "name")），
        # 这里仅处理 keyword arguments。如果有位置参数，可以根据需求扩展。

        return {
            "name": func_name,
            "arguments": arguments
        }

    except SyntaxError:
        return {"error": "语法错误，无法解析输入字符串"}
    except Exception as e:
        return {"error": f"解析失败: {str(e)}"}

# --- 测试用例 ---

# 测试 1: 无参数调用
print(json.dumps(parse_tool_call("pwd()"), ensure_ascii=False))
# 输出: {"name": "pwd", "arguments": {}}

# 测试 2: 带参数调用
print(json.dumps(parse_tool_call('find(path=".")'), ensure_ascii=False))
# 输出: {"name": "find", "arguments": {"path": "."}}

# 测试 3: 多个参数，包含复杂字符
print(json.dumps(parse_tool_call('grep(file_name="main.py", pattern="def init")'), ensure_ascii=False))
# 输出: {"name": "grep", "arguments": {"file_name": "main.py", "pattern": "def init"}}

# 测试 4: 参数包含整数和布尔值
print(json.dumps(parse_tool_call('du(human_readable=True, depth=2)'), ensure_ascii=False))
# 输出: {"name": "du", "arguments": {"human_readable": true, "depth": 2}}