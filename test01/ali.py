# -*- coding: utf-8 -*-

import json
import base64
import os
import ssl
import pandas as pd
import pprint

try:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
except ImportError:
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError

context = ssl._create_unverified_context()


def get_img(img_file):
    """将本地图片转成base64编码的字符串，或者直接返回图片链接"""
    # 简单判断是否为图片链接
    if img_file.startswith("http"):
        return img_file
    else:
        with open(os.path.expanduser(img_file), 'rb') as f:  # 以二进制读取本地图片
            data = f.read()
    try:
        encodestr = str(base64.b64encode(data), 'utf-8')
    except TypeError:
        encodestr = base64.b64encode(data)

    return encodestr


def posturl(headers, body):
    """发送请求，获取识别结果"""
    try:
        params = json.dumps(body).encode(encoding='UTF8')
        req = Request(REQUEST_URL, params, headers)
        r = urlopen(req, context=context)
        html = r.read()
        return html.decode("utf8")
    except HTTPError as e:
        print(e.code)
        print(e.read().decode("utf8"))


def request(appcode, img_file, params):
    # 请求参数
    if params is None:
        params = {}
    img = get_img(img_file)
    if img.startswith('http'):  # img 表示图片链接
        params.update({'url': img})
    else:  # img 表示图片base64
        params.update({'img': img})

    # 请求头
    headers = {
        'Authorization': 'APPCODE %s' % appcode,
        'Content-Type': 'application/json; charset=UTF-8'
    }

    response = posturl(headers, params)
    return json.loads(response)


# 请求接口
REQUEST_URL = "https://gjbsb.market.alicloudapi.com/ocrservice/advanced"

# 配置信息
APPCODE = "e092e15c54924d47986e9f1f09e8d08e"
IMAGE_PATH = "t1.jpg"  # 替换为你的图片路径
params = {
    "prob": False,
    "charInfo": False,
    "rotate": False,
    "table": True,  # 启用表格识别功能
    "sortPage": False,
    "noStamp": False,
    "figure": False,
    "row": False,
    "paragraph": False,
    "oricoord": True
}

result = request(APPCODE, IMAGE_PATH, params)


def extract_table_structure(ocr_result):
    """提取表格结构"""
    table_data = []

    # 检查是否有 prism_tablesInfo 字段
    if "prism_tablesInfo" in ocr_result and ocr_result["prism_tablesInfo"]:
        # 直接从 cellInfos 提取数据，按行组织
        cell_infos = ocr_result["prism_tablesInfo"][0]["cellInfos"]

        # 按照行列坐标组织数据
        rows_data = {}
        for cell in cell_infos:
            row_idx = cell.get("ysc", 0)
            col_idx = cell.get("xsc", 0)
            text = cell.get("word", "")

            if row_idx not in rows_data:
                rows_data[row_idx] = {}
            rows_data[row_idx][col_idx] = text

        # 转换为有序列表
        if rows_data:
            max_row = max(rows_data.keys())
            max_col = max(max(row.keys()) if row else 0 for row in rows_data.values())

            for row_idx in range(max_row + 1):
                row_data = []
                for col_idx in range(max_col + 1):
                    row_data.append(rows_data.get(row_idx, {}).get(col_idx, ""))
                table_data.append(row_data)

    return table_data


def create_dataframe(table_data):
    """创建 DataFrame 来表示表格"""
    df = pd.DataFrame(table_data)
    return df


if __name__ == "__main__":
    if result:
        print("📄 OCR 识别结果（简洁美观输出）")
        print("──────────────────────────────────────")

        table_data = extract_table_structure(result)
        df = create_dataframe(table_data)

        # 打印 DataFrame
        print(df.to_string(index=False, header=False))
        #pprint.pprint(result)
        print("──────────────────────────────────────")

        # 保存为文件
        df.to_csv("ocr_result_clean.csv", index=False, header=False, encoding="utf-8-sig")
        print("✅ 识别结果（简洁输出）已保存至 ocr_result_clean.csv")
    else:
        print("❌ 未识别出任何内容，请检查图片或网络连接。")
