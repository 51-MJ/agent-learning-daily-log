# tool_router.py
# 功能：统一工具注册、关键词匹配、批量调用路由模块
from datetime import datetime

# ---------------------- 定义所有工具函数 ----------------------
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

# ---------------------- 工具注册表：关键词映射工具 ----------------------
tool_registry = [
    {
        "keywords": ["几点","当前时间","现在时间"],
        "func": get_now_time,
        "desc": "获取当前时分秒"
    },
    {
        "keywords": ["几号","日期","今天几号"],
        "func": get_today_date,
        "desc": "获取年月日"
    },
    {
        "keywords": ["星期几","周几","星期"],
        "func": get_today_week,
        "desc": "获取星期"
    }
]

# ---------------------- 批量匹配工具 ----------------------
def match_all_tools(user_input: str) -> list:
    """根据用户输入，匹配所有命中关键词的工具"""
    hit_tools = []
    lower_text = user_input.lower()
    for tool_item in tool_registry:
        for word in tool_item["keywords"]:
            if word in lower_text:
                hit_tools.append(tool_item["func"])
                break
    return hit_tools

# ---------------------- 批量执行工具，汇总结果 ----------------------
def run_all_matched_tools(user_input: str) -> str:
    """匹配并执行全部工具，拼接简洁结果文本"""
    hit_funcs = match_all_tools(user_input)
    if not hit_funcs:
        return ""
    res_list = []
    for fn in hit_funcs:
        res_list.append(fn())
    return " ".join(res_list)