import os
import paddle
from paddleocr import PPStructureV3
from PIL import Image
import json
import pandas as pd
import numpy as np
import time
import psutil


class TableOCRProcessor:
    def __init__(self):
        """初始化OCR处理器"""
        self._print_system_info()
        self._init_paddle()
        self.model = self._init_model()

    def _print_system_info(self):
        """打印系统和硬件信息"""
        print("\n🖥️ 系统信息:")
        print(f"CPU核心数: {os.cpu_count()}")
        print(f"系统内存: {psutil.virtual_memory().total / 1024 ** 3:.2f} GB")
        if paddle.is_compiled_with_cuda():
            try:
                props = paddle.device.cuda.get_device_properties()
                print(f"GPU型号: {props.name}")
                print(f"GPU显存: {props.total_memory / 1024 ** 3:.2f} GB")
            except Exception as e:
                print(f"获取GPU信息失败: {str(e)}")

    def _init_paddle(self):
        """配置PaddlePaddle参数"""
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
            print("\n✅ 已启用GPU加速")
        else:
            paddle.set_device('cpu')
            print("\n⚠️ 未检测到GPU，使用CPU模式")

    def _init_model(self):
        """初始化PPStructureV3模型"""
        print("\n🔥 模型初始化中...")
        start_time = time.time()

        # 初始化模型（不传递任何参数）
        model = PPStructureV3()

        # 小型预热
        try:
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            model.predict(dummy_img)
        except Exception as e:
            print(f"预热失败: {str(e)}")

        print(f"模型初始化完成，耗时: {time.time() - start_time:.2f}秒")
        return model

    def preprocess_image(self, image_path, max_size=1200):
        """
        图像预处理
        修复RGBA转JPEG问题并优化缩放逻辑
        """
        print("\n🖼️ 图像预处理中...")
        try:
            img = Image.open(image_path)

            # 转换RGBA为RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                print("已转换RGBA图像为RGB格式")

            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                print(f"图像从 ({w},{h}) 缩放至 ({new_w},{new_h})")

            return img
        except Exception as e:
            print(f"❌ 图像预处理失败: {str(e)}")
            raise

    def run_structure_ocr(self, image_path):
        """执行OCR识别"""
        print("\n🔍 OCR识别开始")
        start_time = time.time()

        try:
            # 预处理图像
            img = self.preprocess_image(image_path)

            # 使用临时文件（确保RGB格式）
            temp_img_path = "temp_preprocessed.jpg"
            img.save(temp_img_path, quality=95, subsampling=0)

            print(f"当前工作目录: {os.getcwd()}")
            print(f"处理图像尺寸: {img.width}x{img.height}")

            # 执行OCR
            ocr_start = time.time()
            result = self.model.predict(temp_img_path)
            print(f"OCR核心处理耗时: {time.time() - ocr_start:.2f}秒")

            # 保存结果
            output_json_path = "raw_table_output.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n✅ OCR识别完成，总耗时: {time.time() - start_time:.2f}秒")
            return output_json_path
        except Exception as e:
            print(f"❌ OCR识别失败: {str(e)}")
            raise
        finally:
            # 清理临时文件
            if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

    @staticmethod
    def parse_poly_str(poly_str):
        """解析坐标字符串"""
        cleaned = ''.join(c if c.isdigit() or c.isspace() else ' ' for c in poly_str)
        points = np.fromstring(cleaned, sep=' ', dtype=int).reshape(-1, 2)
        return points.tolist()

    @staticmethod
    def cluster_rows(boxes, threshold=15):
        """优化行聚类"""
        top_lefts = np.array([box[0] for box in boxes])
        y_coords = top_lefts[:, 1]
        sorted_idx = np.argsort(y_coords)
        diffs = np.diff(y_coords[sorted_idx])
        split_points = np.where(diffs > threshold)[0] + 1
        return np.split(sorted_idx, split_points)

    def reconstruct_table(self, rec_texts, dt_polys):
        """重建表格"""
        print("\n📊 表格重建中...")
        start_time = time.time()

        try:
            boxes = [self.parse_poly_str(p) for p in dt_polys]
            rows_indices = self.cluster_rows(boxes)
            top_lefts = np.array([box[0] for box in boxes])

            rows = []
            for row in rows_indices:
                row_sorted = sorted(row, key=lambda i: top_lefts[i, 0])
                rows.append([rec_texts[i] for i in row_sorted])

            max_cols = max(len(row) for row in rows)
            df = pd.DataFrame([row + [''] * (max_cols - len(row)) for row in rows])

            print(f"✅ 表格重建完成，耗时: {time.time() - start_time:.2f}秒")
            return df
        except Exception as e:
            print(f"❌ 表格重建失败: {str(e)}")
            raise


def main():
    # 总计时
    total_start = time.time()
    print("\n" + "=" * 50)
    print("🛠️ 表格OCR处理程序启动")
    print("=" * 50)

    # 输入图像
    image_path = "t.jpg"
    if not os.path.exists(image_path):
        print(f"\n❌ 图片文件不存在: {image_path}")
        return

    try:
        # 初始化
        init_start = time.time()
        processor = TableOCRProcessor()
        print(f"\n🔄 初始化总耗时: {time.time() - init_start:.2f}秒")

        # OCR识别
        ocr_start = time.time()
        json_path = processor.run_structure_ocr(image_path)
        print(f"\n🔄 OCR总耗时: {time.time() - ocr_start:.2f}秒")

        # 读取结果
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 验证数据
        if not data or not isinstance(data, list):
            raise ValueError("识别结果格式异常")

        overall_ocr_res = data[0].get('overall_ocr_res')
        if not overall_ocr_res:
            raise ValueError("未找到OCR结果")

        rec_texts = overall_ocr_res.get('rec_texts', [])
        dt_polys = overall_ocr_res.get('dt_polys', [])

        if len(rec_texts) != len(dt_polys):
            raise ValueError("文本与坐标数量不匹配")

        # 表格重建
        rebuild_start = time.time()
        df = processor.reconstruct_table(rec_texts, dt_polys)
        print(f"\n🔄 表格重建总耗时: {time.time() - rebuild_start:.2f}秒")

        # 保存结果
        csv_path = "simple_reconstructed_table.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 结果预览
        print("\n📄 表格预览:")
        print(df.head())
        print(f"\n✅ 结果已保存至: {csv_path}")

    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        return

    # 总耗时
    total_time = time.time() - total_start
    print("\n" + "=" * 50)
    print(f"🏁 全部处理完成！总耗时: {total_time:.2f}秒")
    print("=" * 50)


if __name__ == "__main__":
    try:
        paddle.set_default_dtype('float16')
        print("\n🔼 已启用float16混合精度")
    except:
        print("\n🔽 当前环境不支持float16，使用默认精度")

    main()