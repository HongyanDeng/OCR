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
IMAGE_PATH = "wf2.jpg"  # 替换为你的图片路径

def is_continuous(prev_item, curr_item, x_threshold=30, overlap_threshold=0.5):
    """
    判断两个文字块是否属于同一段连续文字
    :param prev_item: 前一个文字块
    :param curr_item: 当前文字块
    :param x_threshold: 横向距离阈值
    :param overlap_threshold: 高度重叠比例阈值
    :return: 是否连续
    """
    prev_rect = prev_item.get("rect", {})
    curr_rect = curr_item.get("rect", {})

    prev_right = prev_rect.get("left", 0) + prev_rect.get("width", 0)
    curr_left = curr_rect.get("left", 0)

    # 判断是否在同一行（top接近）
    same_line = abs(prev_rect.get("top", 0) - curr_rect.get("top", 0)) < 15

    # 判断是否水平连续
    close_enough = curr_left - prev_right < x_threshold

    return same_line and close_enough


def merge_continuous_words(line):
    """
    合并一行中连续的文字块
    :param line: 按行排序后的文字块列表
    :return: 合并后的段落字符串
    """
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

def group_by_line(ocr_result, y_threshold=15):
    """
    根据 top 坐标对文字块进行分行分组
    :param ocr_result: OCR 识别结果列表
    :param y_threshold: 判断是否为同一行的 y 坐标差值阈值
    :return: 按行分组的结果 [[word1, word2], [word3, word4], ...]
    """
    # 按 top 坐标排序
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


def sort_line_by_x(line, x_threshold=10):
    """
    对每一行中的文字块按 left 坐标排序
    """
    return sorted(line, key=lambda x: x.get("rect", {}).get("left", 0))


def format_line(line):
    """
    格式化一行中的多个文字块，保持空格分隔或合并为段落
    """
    line = sort_line_by_x(line)

    # 如果是连续段落风格（如长句），合并为整段输出
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
