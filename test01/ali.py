from paddleocr import PaddleOCR
import pandas as pd
from bs4 import BeautifulSoup
import os
import paddle
from PIL import Image

# 检查是否使用 GPU
print(paddle.is_compiled_with_cuda())  # 应输出 True

# 初始化 PaddleOCR
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='ch'
)

# 图片路径
image_path = "t1.jpg"
print(os.path.exists(image_path))  # 应输出 True
img = Image.open(image_path)
img.show()  # 查看图片是否正常

# 使用 predict 方法进行识别
result = ocr.predict(image_path)

# 提取表格 HTML（如果识别到表格）
table_html = ""
for line in result:
    if isinstance(line, dict) and 'html' in line:
        table_html = line['html']
        break

# 如果识别到表格，解析 HTML 并转换为 DataFrame
if table_html:
    print("✅ 识别到表格")
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')

    table_data = []
    for row in rows:
        cols = row.find_all(['td', 'th'])
        cols = [col.get_text(strip=True) for col in cols]
        table_data.append(cols)

    # 转换为 DataFrame
    df = pd.DataFrame(table_data[1:], columns=table_data[0])

    # 打印并保存结果
    print("📄 表格识别结果：")
    print(df.to_string(index=False))
    df.to_csv("ocr_result_paddle.csv", index=False, encoding="utf-8-sig")
    print("✅ 表格识别结果已保存至 ocr_result_paddle.csv")
else:
    print("⚠️ 未识别到表格，尝试提取普通文本")
    # 从 result 中提取 rec_texts
    text_result = []
    for line in result:
        if isinstance(line, dict) and 'rec_texts' in line:
            text_result = line['rec_texts']
            break

    if text_result:
        for text in text_result:
            print(text)
        with open("ocr_result_paddle.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(text_result))
        print("✅ 普通文本识别结果已保存至 ocr_result_paddle.txt")
    else:
        print("❌ 未提取到任何文本内容")
