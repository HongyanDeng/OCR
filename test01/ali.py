from paddleocr import PPStructureV3
import pandas as pd
import os
import paddle
from PIL import Image

# 检查是否使用 GPU
print(paddle.is_compiled_with_cuda())  # 应输出 True

# 初始化 PPStructureV3（用于结构化识别）
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

# 提取文本内容
text_result = []
for res in result:
    if isinstance(res, dict) and 'overall_ocr_res' in res:
        ocr_res = res['overall_ocr_res']
        if 'rec_texts' in ocr_res:
            text_result.extend(ocr_res['rec_texts'])

# 如果提取到文本内容
if text_result:
    print(" 识别到文本内容：")
    for text in text_result:
        print(text)

    # 保存原始文本到文件
    with open("ocr_result_paddle.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(text_result))
    print(" 普通文本识别结果已保存至 ocr_result_paddle.txt")

    # 自动组织成表格结构
    # 假设前4个元素是表头（日期、上午、下午、晚上）
    if len(text_result) >= 4:
        # 表头
        headers = text_result[:4]

        # 数据行（跳过表头，每4个元素为一行）
        data_rows = []
        for i in range(4, len(text_result), 4):
            row = text_result[i:i + 4]
            # 如果最后一行不足4列，用空字符串填充
            while len(row) < 4:
                row.append("")
            data_rows.append(row)

        # 创建 DataFrame
        df = pd.DataFrame(data_rows, columns=headers)

        # 打印表格
        print("\n 自动整理的表格识别结果：")
        print(df.to_string(index=False))

        # 保存为 CSV
        df.to_csv("ocr_result_paddle.csv", index=False, encoding="utf-8-sig")
        print(" 表格识别结果已保存至 ocr_result_paddle.csv")
    else:
        print(" 文本内容不足以构成表格结构")
else:
    print(" 未提取到任何文本内容")
