import base64
import json
import os
import ssl
import requests

# 忽略 SSL 证书验证（用于测试环境）
ssl._create_default_https_context = ssl._create_unverified_context

# API 配置
REQUEST_URL = "https://tysbgpu.market.alicloudapi.com/api/predict/ocr_general"
APPCODE = "e092e15c54924d47986e9f1f09e8d08e"
APP_KEY = "204920110"
APP_SECRET = "xAH7EMiguRN4io29tc9E38C0RUNYfrf0"

# 图片路径（支持本地路径或 URL）
IMAGE_PATH = "wf.jpg"  # 替换为你的图片路径


def image_to_base64(img_path):
    """将本地图片转为 base64 编码"""
    if img_path.startswith("http"):
        return img_path  # 如果是 URL，直接返回
    with open(img_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_str


def send_ocr_request():
    """发送 OCR 请求并返回识别结果"""
    image_data = image_to_base64(IMAGE_PATH)

    headers = {
        "Authorization": "APPCODE " + APPCODE,
        "Content-Type": "application/json",
        "X-Ca-Key": APP_KEY,
        "X-Ca-Secret": APP_SECRET,
    }

    payload = {
        "image": image_data,
        "configure": {
            "min_size": 16,
            "output_prob": True,
            "output_keypoints": False,
            "skip_detection": False,
            "without_predicting_direction": False,
        }
    }

    try:
        response = requests.post(REQUEST_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()
        else:
            print("请求失败，状态码：", response.status_code)
            print("错误信息：", response.text)
            return None
    except Exception as e:
        print("请求异常：", e)
        return None


if __name__ == "__main__":
    result = send_ocr_request()
    if result:
        print("识别结果：")
        for item in result.get("result", []):
            print(item.get("text"))
