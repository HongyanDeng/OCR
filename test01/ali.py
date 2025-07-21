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
    """发送 OCR 请求并返回结构化识别结果"""
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
            "output_keypoints": True,
            "skip_detection": False,
            "without_predicting_direction": False,
        }
    }

    try:
        response = requests.post(REQUEST_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return data.get("ret", [])
            else:
                print("API 返回失败，错误信息：", data)
                return []
        else:
            print("请求失败，状态码：", response.status_code)
            print("错误信息：", response.text)
            return []
    except Exception as e:
        print("请求异常：", e)
        return []


def extract_words(ocr_result):
    """提取所有识别出的文字内容"""
    return [item.get("word") for item in ocr_result if "word" in item]


def extract_words_with_prob(ocr_result):
    """提取文字内容及置信度"""
    return [(item.get("word"), item.get("prob")) for item in ocr_result if "word" in item]


def filter_by_prob(ocr_result, threshold=0.9):
    """根据置信度过滤识别结果"""
    return [item for item in ocr_result if item.get("prob", 0) >= threshold]


if __name__ == "__main__":
    result = send_ocr_request()

    if result:
        print("✅ 识别结果：")
        for word in extract_words(result):
            print(word)

        print("\n✅ 识别结果（含置信度）：")
        for word, prob in extract_words_with_prob(result):
            print(f"{word} ({prob:.4f})")

        print("\n✅ 高置信度结果（>0.9）：")
        high_confidence = filter_by_prob(result, threshold=0.9)
        for item in high_confidence:
            print(f"{item['word']} ({item['prob']:.4f})")
    else:
        print("❌ 未识别出任何内容，请检查图片或网络连接。")
