import json
import numpy as np
import pandas as pd

def parse_poly_str(poly_str):
    """
    把类似 "[[106  57]\n ...\n [106  88]]" 形式的字符串解析成4个点的坐标列表[[x,y],...]
    """
    # 去除多余字符，只留下数字和空格
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
    # 读取 OCR 结果 JSON
    with open("raw_table_output.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 你的 OCR 结果在列表第一项的 overall_ocr_res 下
    overall_ocr_res = data[0]['overall_ocr_res']

    rec_texts = overall_ocr_res['rec_texts']
    dt_polys = overall_ocr_res['dt_polys']

    # 重建表格
    df = reconstruct_table(rec_texts, dt_polys)

    print("简易表格重建结果预览：")
    print(df)

    # 保存
    df.to_csv("simple_reconstructed_table.csv", index=False, encoding="utf-8-sig")
    print("✅ 简易表格已保存到 simple_reconstructed_table.csv")

if __name__ == "__main__":
    main()
