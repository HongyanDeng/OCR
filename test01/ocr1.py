import pytesseract
from PIL import Image

# 指定 Tesseract OCR 的路径
pytesseract.pytesseract.tesseract_cmd = r'E:\software\Tesseract-OCR\tesseract.exe'  # 根据实际情况修改路径

# 打开图片文件
image = Image.open('wf2.jpg')

# 使用 pytesseract 进行 OCR
text = pytesseract.image_to_string(image, lang='chi_sim')  # 'chi_sim' 表示使用简体中文模型

# 打印识别出来的文字
print(text)

# 如果需要将结果保存到文件
with open('recognized_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)
