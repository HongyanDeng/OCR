import os
import paddle
from paddleocr import PPStructureV3
from PIL import Image
import json
import pandas as pd
import numpy as np

def run_structure_ocr(image_path):
    print("当前工作目录:", os.getcwd())
    print("是否使用GPU:", paddle.is_compiled_with_cuda())

    ocr = PPStructureV3(use_textline_orientation=True)
    print(f"开始结构化识别图片: {image_path}")
    result = ocr.predict(image_path)

    output_json_path = "raw_table_output.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"结构化识别完成，结果已保存: {output_json_path}")
    return output_json_path

def parse_poly_str(poly_str):
    """
    把类似 "[[106  57]\n ...\n [106  88]]" 形式的字符串解析成4个点的坐标列表[[x,y],...]
    """
    lines = poly_str.strip().replace('[','').replace(']','').split('\n')
    points = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            x, y = map(int, parts)
            points.append([x,y])
    return points

def cluster_rows(boxes, threshold=15):
    """
    根据每个文字框左上点的y坐标聚类成行
    boxes格式：List[List[x,y,...]] 4个顶点坐标
    返回：List[List[int]] 每个子list是同一行的索引
    """
    y_coords = [box[0][1] for box in boxes]  # 取左上角点的y
    sorted_idx = np.argsort(y_coords)
    rows = []
    current_row = []
    last_y = None
    for idx in sorted_idx:
        y = y_coords[idx]
        if last_y is None or abs(y - last_y) <= threshold:
            current_row.append(idx)
        else:
            rows.append(current_row)
            current_row = [idx]
        last_y = y
    if current_row:
        rows.append(current_row)
    return rows

def reconstruct_table(rec_texts, dt_polys):
    # 解析字符串坐标为数字坐标
    boxes = [parse_poly_str(p) for p in dt_polys]

    # 行聚类
    rows_indices = cluster_rows(boxes, threshold=15)

    # 组装行文本（行内根据左上点x排序）
    rows = []
    for row in rows_indices:
        # 行内排序
        row_sorted = sorted(row, key=lambda i: boxes[i][0][0])
        row_texts = [rec_texts[i] for i in row_sorted]
        rows.append(row_texts)

    # 转成DataFrame
    df = pd.DataFrame(rows)
    return df

def main():
    image_path = "t.jpg"

    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return

    # 运行结构化识别，保存结果json
    json_path = run_structure_ocr(image_path)

    if not os.path.exists(json_path):
        print(f"❌ 识别结果文件未生成: {json_path}")
        return

    # 读取识别结果json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data通常是列表，第一项为本页结果
    if not data or not isinstance(data, list):
        print("❌ 识别结果格式异常")
        return

    overall_ocr_res = data[0].get('overall_ocr_res', None)
    if not overall_ocr_res:
        print("❌ 未找到 overall_ocr_res 字段")
        return

    rec_texts = overall_ocr_res.get('rec_texts', [])
    dt_polys = overall_ocr_res.get('dt_polys', [])

    if not rec_texts or not dt_polys or len(rec_texts) != len(dt_polys):
        print("❌ 文本与坐标数量不匹配或为空")
        return

    # 重建简易表格
    df = reconstruct_table(rec_texts, dt_polys)

    print("\n📄 简易表格重建结果预览：")
    print(df)

    csv_path = "simple_reconstructed_table.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 简易表格已保存到: {csv_path}")

if __name__ == "__main__":
    main()
