import ast
import json

def convert_python_to_xml(input_str):
    """
    解析 Python 函数调用格式的字符串，转换为 <tool_call> XML 格式。
    不执行代码，仅做静态语法解析。
    """
    result_parts = []
    
    try:
        # 1. 使用 ast 解析字符串结构 (mode='eval' 表示解析一个表达式)
        tree = ast.parse(input_str, mode='eval')
        
        # 2. 确保最外层是一个列表 [...]
        if not isinstance(tree.body, ast.List):
            return "Error: Input string must be a list [ ... ]"
            
        # 3. 遍历列表中的每一个元素
        for node in tree.body.elts:
            # 确保元素是一个函数调用，例如 cd(...)
            if isinstance(node, ast.Call):
                # --- A. 提取函数名 ---
                # node.func.id 就是函数名 (如 'cd', 'cat')
                func_name = node.func.id
                
                # --- B. 提取关键字参数 ---
                # node.keywords 包含所有 key=value 形式的参数
                args_dict = {}
                for keyword in node.keywords:
                    key = keyword.arg  # 参数名，如 'folder'
                    # keyword.value 是 AST 节点，我们需要它的字面值
                    # literal_eval 安全地将 AST 节点转为 Python 基本类型 (str, int, bool)
                    value = ast.literal_eval(keyword.value)
                    args_dict[key] = value
                
                # --- C. 构建 JSON 和 XML ---
                json_args = json.dumps(args_dict, ensure_ascii=False)
                
                xml_block = (
                    f'<tool_call>\n'
                    f'{{"name": "{func_name}", "arguments": {json_args}}}\n'
                    f'</tool_call>'
                )
                result_parts.append(xml_block)
                
    except SyntaxError as e:
        return f"解析错误: 输入的字符串不符合 Python 语法 format. ({e})"
    except Exception as e:
        return f"转换过程中发生错误: {e}"

    return "\n".join(result_parts)

# --- 测试 ---
if __name__ == '__main__':
    # 这是一个纯字符串，不是代码执行
    input_text = '[cd(folder="data"), cat(file_name="unggv.json"), cd(folder="raw"), tail(file_name="dtpsp.json", lines=7)]'

    # 转换
    output_text = convert_python_to_xml(input_text)
    print(output_text)