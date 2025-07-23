import base64
import json
import os
import ssl
import requests
import pandas as pd

# 忽略 SSL 证书验证（用于测试环境）
ssl._create_default_https_context = ssl._create_unverified_context

# API 配置
REQUEST_URL = "https://tysbgpu.market.alicloudapi.com/api/predict/ocr_general"
APPCODE = "e092e15c54924d47986e9f1f09e8d08e"
APP_KEY = "204920110"
APP_SECRET = "xAH7EMiguRN4io29tc9E38C0RUNYfrf0"

# 图片路径（支持本地路径或 URL）
IMAGE_PATH = "t1.jpg"  # 替换为你的图片路径

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

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

def extract_table_structure(ocr_result):
    """提取表格结构"""
    table_data = []

    for item in ocr_result:
        word = item.get("word", "")
        rect = item.get("rect", {})
        table_data.append({
            "text": word,
            "left": rect.get("left", 0),
            "top": rect.get("top", 0),
            "width": rect.get("width", 0),
            "height": rect.get("height", 0)
        })

    return table_data

def group_by_row(table_data, y_threshold=15):
    """根据 top 坐标对文字块进行分行分组"""
    sorted_data = sorted(table_data, key=lambda x: (x["top"], x["left"]))

    rows = []
    current_row = []
    last_top = None

    for item in sorted_data:
        current_top = item["top"]

        if last_top is None or abs(current_top - last_top) <= y_threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]

        last_top = current_top

    if current_row:
        rows.append(current_row)

    return rows

def format_row(row):
    """格式化一行中的多个文字块"""
    sorted_row = sorted(row, key=lambda x: x["left"])
    return " ".join([item["text"] for item in sorted_row])

def format_table(rows):
    """格式化整个表格"""
    formatted_rows = [format_row(row) for row in rows]
    return "\n".join(formatted_rows)

def create_dataframe(table_data):
    """创建 DataFrame 来表示表格"""
    rows = group_by_row(table_data)
    data = []

    for row in rows:
        formatted_row = format_row(row)
        data.append(formatted_row.split())

    # 确保每行具有相同的列数
    max_cols = max(len(row) for row in data)
    for row in data:
        while len(row) < max_cols:
            row.append("")

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    result = send_ocr_request()

    if result:
        print("📄 OCR 识别结果（简洁美观输出）")
        print("──────────────────────────────────────")

        table_data = extract_table_structure(result)
        df = create_dataframe(table_data)

        # 打印 DataFrame
        print(df.to_string(index=False, header=False))

        print("──────────────────────────────────────")

        # 保存为文件
        df.to_csv("ocr_result_clean.csv", index=False, header=False, encoding="utf-8-sig")
        print("✅ 识别结果（简洁输出）已保存至 ocr_result_clean.csv")
    else:
        print("❌ 未识别出任何内容，请检查图片或网络连接。")
