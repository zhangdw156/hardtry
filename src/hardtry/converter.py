import ast
import json

def _get_node_name(node):
    """
    递归提取复杂的函数名。
    支持: print, os.path.join, tools['search'], get_tool()
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_get_node_name(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        value = _get_node_name(node.value)
        slice_val = "?"
        if isinstance(node.slice, ast.Constant):
            slice_val = repr(node.slice.value)
        else:
            slice_val = ast.unparse(node.slice)
        return f"{value}[{slice_val}]"
    elif isinstance(node, ast.Call):
        return ast.unparse(node)
    else:
        return ast.unparse(node)

def _get_arg_value(node):
    """
    解析参数值。
    策略：优先转为 Python 原生对象 (int, str, list...)，
    如果遇到变量或表达式 (如 x+1, func())，则回退为源码字符串。
    """
    try:
        # 尝试把 AST 节点转为 Python 对象 (例如: "hello", 123, [1, 2])
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        # 如果包含变量、函数调用或运算 (例如: x, 1+1, call())
        # 使用 unparse 还原为代码字符串
        return ast.unparse(node)

def convert_python_to_xml(input_str:str)->list[dict[str,str]]:
    """
    将 Python 函数调用列表转换为 <tool_call> XML 格式。
    静态解析，安全，不执行代码。
    """
    result_parts = []
    
    try:
        # 1. 解析模式：eval (处理表达式)
        tree = ast.parse(input_str.strip(), mode='eval')
        
        # 2. 校验最外层是否为列表
        if not isinstance(tree.body, ast.List):
            # 容错处理：如果用户没有包 [], 尝试把它当做单个 Call 处理?
            # 这里为了严谨，我们坚持要求是列表，或者你可以扩展逻辑
            return "Error: Input must be a list of calls, e.g., [func1(), func2()]"
            
        # 3. 遍历列表元素
        for node in tree.body.elts:
            # 只处理函数调用
            if isinstance(node, ast.Call):
                # --- A. 获取全能函数名 ---
                func_name = _get_node_name(node.func)
                
                args_dict = {}
                
                # --- B. 处理位置参数 (Positional Args) ---
                for i, arg in enumerate(node.args):
                    args_dict[f"arg_{i}"] = _get_arg_value(arg)
                
                # --- C. 处理关键字参数 (Keyword Args) ---
                for keyword in node.keywords:
                    key = keyword.arg
                    value = _get_arg_value(keyword.value)
                    args_dict[key] = value
                
                # --- D. 构建 json ---
                # 使用 ensure_ascii=False 保证中文不乱码
                json_args = json.dumps(args_dict, ensure_ascii=False)
                
                json_block :dict= {"name": func_name, "arguments": json_args}
                result_parts.append(json_block)
            else:
                # 如果列表里混入了非函数调用的东西（比如 [1, 2]），选择跳过或报警
                print(f"Warning: Skipped non-call element: {ast.unparse(node)}")

    except SyntaxError as e:
        return f"SyntaxError: 输入的代码不符合 Python 语法。详情: {e}"
    except Exception as e:
        return f"SystemError: 转换过程发生未知错误: {e}"

    return result_parts

# --- 各种刁钻的测试用例 ---
if __name__ == '__main__':
    test_case = """
[
    cd(folder="documents"), 
    os.path.join(path="src", file="main.py"), 
    tools['search'](query="AI Agents"), 
    tools["search"](query="AI Agents"), 
    calculator(1, 2, mode="add"), 
    complex_func(param=x_variable, param2=1+1, callback=lambda x: x)
]
"""
    
    print("--- 输入 ---")
    print(test_case.strip())
    print("\n--- 输出 ---")
    print(convert_python_to_xml(test_case))