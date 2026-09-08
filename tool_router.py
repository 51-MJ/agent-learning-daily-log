# tool_router.py
# 功能：统一工具注册、关键词匹配、批量调用路由模块（仅处理时间工具）
from datetime import datetime
from duckduckgo_search import DDGS

# ---------------------- 定义所有时间工具函数 ----------------------
def get_now_time() -> str:
    """获取当前时分秒"""
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def get_today_date() -> str:
    """获取年月日日期"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日")

def get_today_week() -> str:
    """获取星期"""
    week_map = {0:"星期一",1:"星期二",2:"星期三",3:"星期四",4:"星期五",5:"星期六",6:"星期日"}
    week_idx = datetime.now().weekday()
    return week_map[week_idx]

# ---------------------- 计算器工具 ----------------------
def calculate(expression: str) -> str:
    """
    执行数学计算
    :param expression: 数学表达式字符串，如 "1+2*3"、"(5+3)*2"
    :return: 计算结果文本
    """
    try:
        # 只允许数字和基本运算符，防止注入
        allowed = set("0123456789+-*/().%")
        if not all(c in allowed for c in expression):
            return "计算失败：表达式包含不支持的字符，仅支持数字和 + - * / ( ) %"
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"

# ---------------------- 联网搜索工具 ----------------------
def web_search(query: str, max_results: int = 3) -> str:
    """
    联网搜索（DuckDuckGo，无需API key）
    :param query: 搜索关键词
    :param max_results: 返回结果数量
    :return: 搜索结果文本
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"搜索结果：未找到关于「{query}」的相关信息"
        lines = [f"联网搜索「{query}」结果："]
        for i, item in enumerate(results, 1):
            title = item.get("title", "")
            body = item.get("body", "")
            lines.append(f"{i}. {title}\n   {body}")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索失败：{e}，请检查网络连接"
    
# ---------------------- 意图分类 ----------------------
def classify_intent(user_input: str) -> str:
    """
    判断用户提问的意图类型
    :return: "time" / "calculate" / "search" / "knowledge"
    """
    text = user_input.lower()

    # 1. 时间意图（优先级最高）
    time_keywords = ["几点", "现在时间", "当前时间", "几号", "日期", "今天几号", "星期几", "周几", "今天星期"]
    for kw in time_keywords:
        if kw in text:
            return "time"

    # 2. 计算意图
    calc_keywords = ["计算", "等于多少", "算一下", "加", "减", "乘", "除", "平方", "根号"]
    for kw in calc_keywords:
        if kw in text:
            return "calculate"
    # 纯数字表达式也判定为计算（如 "1+2*3"）
    calc_chars = set("0123456789+-*/().% ")
    if len(text) > 1 and all(c in calc_chars for c in text) and any(c in "+-*/" for c in text):
        return "calculate"

    # 3. 搜索意图
    search_keywords = ["搜索", "搜一下", "查一下", "百度", "谷歌", "网上", "最新", "新闻", "最近"]
    for kw in search_keywords:
        if kw in text:
            return "search"

    # 4. 默认走知识库问答
    return "knowledge"

# ---------------------- 统一工具执行 ----------------------
def run_tools(user_input: str, intent: str) -> str:
    """
    根据意图执行对应工具，返回格式化结果
    :param user_input: 用户原始提问
    :param intent: 意图类型
    :return: 工具结果文本
    """
    if intent == "time":
        # 时间工具：匹配关键词执行对应函数
        hit = []
        text = user_input.lower()
        if any(kw in text for kw in ["几点", "时间"]):
            hit.append(get_now_time())
        if any(kw in text for kw in ["几号", "日期"]):
            hit.append(get_today_date())
        if any(kw in text for kw in ["星期", "周几"]):
            hit.append(get_today_week())
        return "、".join(hit) if hit else "无时间相关信息"

    if intent == "calculate":
        # 计算器：提取表达式（简单处理，去掉中文前缀）
        expression = user_input
        for prefix in ["计算", "算一下", "等于多少", "请问", "帮我算"]:
            expression = expression.replace(prefix, "")
        expression = expression.strip("？?。，, ")
        return calculate(expression)

    if intent == "search":
        # 联网搜索：提取搜索关键词
        query = user_input
        for prefix in ["搜索", "搜一下", "查一下", "百度", "谷歌", "网上", "帮我"]:
            query = query.replace(prefix, "")
        query = query.strip("？?。，, ")
        return web_search(query)

    return ""

# 保留原有函数名，兼容旧demo
def run_time_tools(user_input: str) -> str:
    """兼容旧版：仅执行时间工具"""
    return run_tools(user_input, "time")

# 导出对外接口
__all__ = [
    "classify_intent",
    "run_tools",
    "run_time_tools",
    "calculate",
    "web_search"
]