from rag_utils import add_knowledge

# 自定义你的知识库文本
knowledge = [
    "马艳是金融行业资深基金分析师，专注指数基金研究，擅长分析宽基指数定投策略。",
    "指数基金跟踪市场指数，分散单一股票风险，适合长期持有定投。"
]
add_knowledge(knowledge)
print("知识库写入完成，向量自动持久化保存")