from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """获取当前时间。当用户询问现在几点、当前时间时使用此工具。"""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。当用户需要进行数学计算时使用此工具。参数 expression 是数学表达式如 '2 + 3 * 4'。"""
    try:
        # 安全地计算表达式（只允许数字和基本运算符）
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "[error]: expression contains invalid characters."
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"[error]: computed error: {e}"


@tool
def search_code_example(language: str, topic: str) -> str:
    """搜索代码示例。当用户需要某个编程语言的代码示例时使用此工具。参数 language 是编程语言如 python，topic 是主题如排序。"""
    # 模拟的代码示例库
    examples = {
        ("python", "排序"): """
# Python 排序示例
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# 方法1: sorted() 返回新列表
sorted_numbers = sorted(numbers)

# 方法2: list.sort() 原地排序
numbers.sort()

# 方法3: 自定义排序
students = [{"name": "Alice", "score": 85}, {"name": "Bob", "score": 92}]
students.sort(key=lambda x: x["score"], reverse=True)
""",
        ("python", "文件读写"): """
# Python 文件读写示例

# 读取文件
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 写入文件
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!")

# 追加内容
with open("log.txt", "a", encoding="utf-8") as f:
    f.write("New log entry\\n")
""",
    }

    key = (language.lower(), topic)
    if key in examples:
        return examples[key]
    return f"[error]: no code example found for {language} about {topic}"


ALL_TOOLS = [get_current_time, calculate, search_code_example]
