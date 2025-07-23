import base64
import json
import os
import ssl
import requests
import numpy as np
import cv2



# 忽略 SSL 证书验证（用于测试环境）
ssl._create_default_https_context = ssl._create_unverified_context

# API 配置
REQUEST_URL = "https://tysbgpu.market.alicloudapi.com/api/predict/ocr_general"
APPCODE = "e092e15c54924d47986e9f1f09e8d08e"
APP_KEY = "204920110"
APP_SECRET = "xAH7EMiguRN4io29tc9E38C0RUNYfrf0"

# 图片路径（支持本地路径或 URL）
IMAGE_PATH = "t1.jpg"  # 替换为你的图片路径



# 在 image_to_base64 函数中使用预处理后的图像
def image_to_base64(img_path):
    processed_img = preprocess_image(img_path)
    _, buffer = cv2.imencode('.jpg', processed_img)
    encoded_str = base64.b64encode(buffer).decode("utf-8")
    return encoded_str


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 去噪
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # 二值化
    _, binary = cv2.threshold(sharpened, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return binary


def is_continuous(prev_item, curr_item, x_threshold=30, overlap_threshold=0.5):
    prev_rect = prev_item.get("rect", {})
    curr_rect = curr_item.get("rect", {})

    prev_right = prev_rect.get("left", 0) + prev_rect.get("width", 0)
    curr_left = curr_rect.get("left", 0)
    prev_bottom = prev_rect.get("top", 0) + prev_rect.get("height", 0)
    curr_top = curr_rect.get("top", 0)

    # 判断是否在同一行（top接近）
    same_line = abs(prev_rect.get("top", 0) - curr_rect.get("top", 0)) < 15

    # 判断是否水平连续
    close_enough = curr_left - prev_right < x_threshold

    # 判断高度是否有重叠
    height_overlap = max(0, min(prev_bottom, curr_top + curr_rect.get("height", 0)) - max(prev_rect.get("top", 0),
                                                                                          curr_top))
    height_overlap_ratio = height_overlap / min(prev_rect.get("height", 0), curr_rect.get("height", 0))

    return same_line and close_enough and height_overlap_ratio >= overlap_threshold


def merge_continuous_words(line):
    if not line:
        return ""

    merged_line = [line[0]["word"]]
    prev_item = line[0]

    for item in line[1:]:
        if is_continuous(prev_item, item):
            merged_line[-1] += item["word"]
        else:
            merged_line.append(item["word"])
        prev_item = item

    return " ".join(merged_line)


def group_by_line(ocr_result, y_threshold=15):
    sorted_result = sorted(ocr_result, key=lambda x: x.get("rect", {}).get("top", 0))

    lines = []
    current_line = []
    last_top = None

    for item in sorted_result:
        current_top = item.get("rect", {}).get("top", 0)

        if last_top is None or abs(current_top - last_top) <= y_threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]

        last_top = current_top

    if current_line:
        lines.append(current_line)

    return lines


def format_line(line):
    line = sort_line_by_x(line)

    if len(line) > 5:  # 字块数量多，可能是段落
        return merge_continuous_words(line)
    else:
        output = ""
        prev_right = None

        for item in line:
            word = item.get("word", "")
            left = item.get("rect", {}).get("left", 0)

            if prev_right is not None and prev_right + 10 < left:
                output += " " * ((left - prev_right) // 8)

            output += word
            prev_right = left + item.get("rect", {}).get("width", 0)

        return output


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



def sort_line_by_x(line, x_threshold=10):
    """
    对每一行中的文字块按 left 坐标排序
    """
    return sorted(line, key=lambda x: x.get("rect", {}).get("left", 0))



if __name__ == "__main__":
    result = send_ocr_request()

    if result:
        print("📄 OCR 识别结果（简洁美观输出）")
        print("──────────────────────────────────────")

        lines = group_by_line(result)

        cleaned_result = []

        for line in lines:
            merged_text = merge_continuous_words(sort_line_by_x(line))

            # 去除空行或纯空白内容
            if merged_text.strip():
                cleaned_result.append(merged_text)

        # 输出结果
        for i, text in enumerate(cleaned_result, 1):
            print(f"{i}. {text}")

        print("──────────────────────────────────────")

        # 保存为文件
        with open("ocr_result_clean.txt", "w", encoding="utf-8") as f:
            for text in cleaned_result:
                f.write(f"{text}\n")
        print("✅ 识别结果（简洁输出）已保存至 ocr_result_clean.txt")
    else:
        print("❌ 未识别出任何内容，请检查图片或网络连接。")
