from paddleocr import PPStructureV3
import pandas as pd
import os
import paddle
from PIL import Image
import json

# 检查是否使用 GPU
print(paddle.is_compiled_with_cuda())  # 应输出 True

# 初始化 PPStructureV3（使用最基本配置）
ocr = PPStructureV3(
    use_textline_orientation=True  # 是否启用文本行方向识别
)

# 图片路径
image_path = "t.jpg"
print(os.path.exists(image_path))  # 应输出 True
img = Image.open(image_path)
print("图片尺寸:", img.size)

# 使用 predict 方法进行识别
result = ocr.predict(image_path)

# 打印完整结果结构（用于调试）
print("识别结果结构:")
print(json.dumps(result, default=str, indent=2)[:2000] + "...")  # 打印更多内容用于分析

# 尝试从结果中提取表格结构信息
table_data = []

print("\n=== 分析识别结果 ===")

# 遍历结果寻找 table_res_list
for i, res in enumerate(result):
    print(f"\n结果项 {i} 类型: {type(res)}")
    if isinstance(res, dict):
        print(f"键值: {list(res.keys())}")

        # 检查是否有 table_res_list
        if 'table_res_list' in res and res['table_res_list']:
            print("✅ 检测到 table_res_list，尝试提取表格结构信息...")

            table_res_list = res['table_res_list']
            for table_res in table_res_list:
                if 'cell_content_list' in table_res and 'cell_box_list' in table_res:
                    cell_contents = table_res['cell_content_list']
                    cell_boxes = table_res['cell_box_list']


                    print("cell_contents 数量：", len(cell_contents))
                    print("cell_boxes 数量：", len(cell_boxes))
                    print("前5个 cell_contents：", cell_contents[:5])
                    print("前5个 cell_boxes：", cell_boxes[:5])

                    print(f"检测到 {len(cell_contents)} 个单元格")

                    if len(cell_contents) == 0 or len(cell_boxes) == 0:
                        print("❌ 单元格内容或坐标为空")
                        continue

                    if len(cell_contents) != len(cell_boxes):
                        print("⚠️ 单元格内容和坐标数量不一致，尝试修复...")

                        # 修复：取较小的数量
                        min_len = min(len(cell_contents), len(cell_boxes))
                        cell_contents = cell_contents[:min_len]
                        cell_boxes = cell_boxes[:min_len]

                    # 尝试根据 cell_boxes 排序并重建表格行
                    # 根据单元格的 y 坐标对单元格进行排序
                    sorted_cells = sorted(
                        zip(cell_contents, cell_boxes),
                        key=lambda x: x[1][0][1]  # 按第一个点的 y 坐标排序
                    )

                    # 根据 x 坐标分组为行
                    rows = []
                    current_row = []
                    prev_y = None
                    row_threshold = 20  # 同一行的 y 差阈值

                    for content, box in sorted_cells:
                        current_y = box[0][1]
                        if prev_y is None or abs(current_y - prev_y) < row_threshold:
                            current_row.append(content)
                        else:
                            rows.append(current_row)
                            current_row = [content]
                        prev_y = current_y

                    if current_row:
                        print("当前构建行：", current_row)

                        rows.append(current_row)

                    # 构建 DataFrame
                    if rows:
                        print("\n📄 成功从结构信息重建表格：")
                        df = pd.DataFrame(rows)
                        print(df.to_string(index=False, header=False))

                        df.to_csv("ocr_structured_table.csv", index=False, encoding="utf-8-sig", header=False)
                        print("✅ 表格已保存至 ocr_structured_table.csv")


                    else:
                        print("❌ 无法构建表格：无有效行")
