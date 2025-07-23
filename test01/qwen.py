import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
)

def call_qwen_vl(image_url, user_text):
    response = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_text}
                ]
            }
        ]
    )
    return response.choices[0].message.content

# 示例调用
if __name__ == "__main__":
    image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
    user_text = "请描述这张图片的内容。"
    result = call_qwen_vl(image_url, user_text)
    print(result)
