from datetime import datetime

# 获取当前年月日
today = datetime.now().strftime("%Y-%m-%d")
# 拼接日志路径
file_path = f"./daily_log/{today}.md"

# 打卡模板内容
md_content = f"""# {today} 学习打卡
### 今日学习内容：

### 遇到的问题 & 踩坑：

### 明日学习计划：
"""

# 创建并写入文件
with open(file_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"✅ 今日打卡文件 {today}.md 生成成功，去daily_log文件夹填写内容即可")