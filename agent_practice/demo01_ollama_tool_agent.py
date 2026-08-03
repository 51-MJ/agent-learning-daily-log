from smolagents import LiteLLMModel, CodeAgent,Tool
import datetime
#1.连接本地ollama模型
#ollama默认地址 http://localhost:11434
model = LiteLLMModel(
    model_id="ollama/qwen2.5:7b",
    api_base="http://localhost:11434",
    api_key="dummy"
)

#2.创建工具
#工具一：获取当前时间
class GetNowTime(Tool):
    name ="get_now_time"
    description="获取当前时间"
    inputs ={}
    output_type = "string"
    def forward(self):
        return str(datetime.datetime.now())

#工具二：简易计算器
class Calculator(Tool):
    name ="calculator"
    description="简易计算器，支持加减乘除运算"
    inputs ={
        "a":{"type":"number","description":"第一个数字"},
        "b":{"type":"number","description":"第二个数字"}
    }
    output_type = "string"
    def forward(self,a,b):
        return  f"计算结果：{a + b}"

#3.初始化Agent，注册可用工具
agent= CodeAgent(
    model=model,
    tools=[GetNowTime(),Calculator()],
    add_base_tools=False
)

#4.测试提问
if __name__ == "__main__":
    question1="现在几点了？"
    print("问题:",question1)
    answer1=agent.run(question1)
    print("回答:",answer1)
    question2="计算3.5+4.2的结果"
    print("问题:",question2)
    answer2=agent.run(question2)
    print("回答:",answer2)