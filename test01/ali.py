import os
import paddle
from paddleocr import PPStructureV3
from PIL import Image
import json
import pandas as pd
import numpy as np
import time  # 新增时间模块


class TableOCRProcessor:
    def __init__(self):
        """初始化OCR处理器"""
        self._init_paddle()
        self.model = self._init_model()

    def _init_paddle(self):
        """配置PaddlePaddle的GPU优化参数"""
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
            print("已启用GPU加速")
        else:
            paddle.set_device('cpu')
            print("未检测到GPU，使用CPU模式")

    def _init_model(self):
        """初始化PPStructureV3模型"""
        return PPStructureV3()

    def print_gpu_status(self):
        """打印GPU状态"""
        if paddle.is_compiled_with_cuda():
            try:
                alloc_mem = paddle.device.cuda.max_memory_allocated() / 1024 ** 2
                total_mem = paddle.device.cuda.get_device_properties().total_memory / 1024 ** 2
                print(f"GPU内存使用: {alloc_mem:.2f}MB / {total_mem:.2f}MB")
            except AttributeError:
                print("无法获取详细显存信息，但GPU已启用")

    def preprocess_image(self, image_path, max_size=1600):
        """图像预处理，调整过大图像尺寸"""
        img = Image.open(image_path)
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            print(f"图像已从({w},{h})缩放至({img.width},{img.height})")
        return img

    def run_structure_ocr(self, image_path):
        """执行结构化OCR识别"""
        print("当前工作目录:", os.getcwd())
        print("是否使用GPU:", paddle.is_compiled_with_cuda())
        self.print_gpu_status()

        try:
            print(f"开始结构化识别图片: {image_path}")
            result = self.model.predict(image_path)
        except Exception as e:
            print(f"识别过程中出错: {str(e)}")
            raise
        finally:
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

        output_json_path = "raw_table_output.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"结构化识别完成，结果已保存: {output_json_path}")
        self.print_gpu_status()
        return output_json_path

    @staticmethod
    def parse_poly_str(poly_str):
        """坐标解析函数"""
        cleaned = ''.join(c if c.isdigit() or c.isspace() else ' ' for c in poly_str)
        points = np.fromstring(cleaned, sep=' ', dtype=int).reshape(-1, 2)
        return points.tolist()

    @staticmethod
    def cluster_rows(boxes, threshold=15):
        """行聚类"""
        top_lefts = np.array([box[0] for box in boxes])
        y_coords = top_lefts[:, 1]
        sorted_idx = np.argsort(y_coords)
        diffs = np.diff(y_coords[sorted_idx])
        split_points = np.where(diffs > threshold)[0] + 1
        rows_indices = np.split(sorted_idx, split_points)
        return [row.tolist() for row in rows_indices]

    def reconstruct_table(self, rec_texts, dt_polys):
        """重建表格"""
        boxes = [self.parse_poly_str(p) for p in dt_polys]
        rows_indices = self.cluster_rows(boxes, threshold=15)
        top_lefts = np.array([box[0] for box in boxes])

        rows = []
        for row in rows_indices:
            row_sorted = sorted(row, key=lambda i: top_lefts[i, 0])
            row_texts = [rec_texts[i] for i in row_sorted]
            rows.append(row_texts)

        max_cols = max(len(row) for row in rows)
        padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
        return pd.DataFrame(padded_rows)


def main():
    # 记录开始时间
    start_time = time.time()

    # 处理实际图像
    image_path = "wf.jpg"
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return

    # 初始化处理器
    processor = TableOCRProcessor()

    # 运行结构化识别
    try:
        json_path = processor.run_structure_ocr(image_path)
    except Exception as e:
        print(f"❌ OCR处理失败: {str(e)}")
        return

    if not os.path.exists(json_path):
        print(f"❌ 识别结果文件未生成: {json_path}")
        return

    # 读取识别结果
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 检查数据格式
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

    # 重建表格
    df = processor.reconstruct_table(rec_texts, dt_polys)

    print("\n📄 简易表格重建结果预览：")
    print(df.head())

    csv_path = "simple_reconstructed_table.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 简易表格已保存到: {csv_path}")

    # 计算并打印总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n⏱️ 总运行时间: {total_time:.2f}秒")


if __name__ == "__main__":
    try:
        paddle.set_default_dtype('float16')
        print("已启用float16精度模式")
    except:
        print("当前环境不支持float16模式，使用默认精度")

    main()