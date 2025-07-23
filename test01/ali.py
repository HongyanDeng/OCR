from paddleocr import PPStructureV3
import pandas as pd
from bs4 import BeautifulSoup
import os
import paddle
from PIL import Image

# 检查是否使用 GPU
print(paddle.is_compiled_with_cuda())  # 应输出 True

# 初始化 PPStructureV3（不传 lang）
ocr = PPStructureV3(
    use_textline_orientation=True  # 是否启用文本行方向识别
)

# 图片路径
image_path = "t1.jpg"
print(os.path.exists(image_path))  # 应输出 True
img = Image.open(image_path)
print("图片尺寸:", img.size)
# img.show()  # 查看图片是否正常（可选）

# 使用 predict 方法进行识别 ✅
result = ocr.predict(image_path)

# 打印识别结果结构（调试用）
# import pprint
# pprint.pprint(result, depth=2)

# 提取文本内容
text_result = []
for res in result:
    if isinstance(res, dict) and 'overall_ocr_res' in res:
        ocr_res = res['overall_ocr_res']
        if 'rec_texts' in ocr_res:
            text_result.extend(ocr_res['rec_texts'])

# 如果提取到文本内容
if text_result:
    print("📄 识别到文本内容：")
    for text in text_result:
        print(text)

    # 保存到文件
    with open("ocr_result_paddle.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(text_result))
    print("✅ 普通文本识别结果已保存至 ocr_result_paddle.txt")
else:
    print("❌ 未提取到任何文本内容")
