# demo22_multimodal_vision.py
# 功能：多模态图片理解，上传图片让模型看，根据图片内容回答问题
# 前置Python知识点：
#   1. base64编码：把图片文件转成文本格式，才能通过API传给模型
#   2. OLLAMA多模态API：messages里的content可以是列表，包含文字和图片
#   3. gr.Image：Gradio的图片上传组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import json
import base64
import gradio as gr

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "minicpm-v:8b"
  # 注意：换成视觉版模型


def image_to_base64(image_path: str) -> str:
    """把图片文件转成base64编码"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def chat_with_image(user_input: str, image_file):
    """
    带图片的对话
    :param user_input: 用户问题
    :param image_file: Gradio上传的图片文件（None表示没传图）
    """
    if image_file is not None:
        # 有图片：转base64，用Ollama的格式传images字段
        img_b64 = image_to_base64(image_file)
        messages = [
            {
                "role": "user",
                "content": f"请用中文回答：{user_input}",
                "images": [img_b64]
 # 注意：Ollama格式，直接在message里加images字段
            }
        ]
    else:
        # 没图片：纯文字对话
        messages = [
            {"role": "user", "content": user_input}
        ]

    payload = {"model": MODEL_NAME, "messages": messages, "stream": True}
    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    full_answer = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "message" in data:
                text = data["message"].get("content", "")
                full_answer += text
                yield full_answer
            if data.get("done", False):
                break


# 启动Gradio界面
with gr.Blocks(title="多模态图片理解") as demo:
    gr.Markdown("# 多模态图片理解助手")
    gr.Markdown("上传一张图片，问关于图片的问题")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="上传图片")
            text_input = gr.Textbox(label="你的问题", placeholder="比如：这张图里有什么？")
            btn = gr.Button("提问", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="回答", lines=10)

    btn.click(
        chat_with_image,
        inputs=[text_input, image_input],
        outputs=[output]
    )

if __name__ == "__main__":
    demo.launch()
