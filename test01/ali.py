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

    def preprocess_image(self, image_path, max_size=3200):
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
        print(f"\n🔍 OCR识别开始: {image_path}")
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

            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            # 确保 ocr_process 文件夹存在
            os.makedirs("ocr_process", exist_ok=True)
            output_json_path = os.path.join("ocr_process", f"{base_name}_table_output.json")

            # 保存结果
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n✅ OCR识别完成，总耗时: {time.time() - start_time:.2f}秒")
            return output_json_path, result
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
    def parse_poly_str(poly):
        """解析坐标字符串或数组"""
        # 如果是 numpy 数组，直接返回tolist()
        if isinstance(poly, np.ndarray):
            return poly.tolist()
        # 如果是列表，直接返回
        if isinstance(poly, list):
            return poly
        # 如果是字符串，按原来的方式处理
        if isinstance(poly, str):
            cleaned = ''.join(c if c.isdigit() or c.isspace() else ' ' for c in poly)
            points = np.fromstring(cleaned, sep=' ', dtype=int).reshape(-1, 2)
            return points.tolist()
        # 其他情况，尝试转换为 numpy 数组
        try:
            return np.array(poly).tolist()
        except:
            raise ValueError(f"无法解析坐标数据: {poly}")

    @staticmethod
    def cluster_rows(boxes, threshold=15):
        """优化行聚类"""
        top_lefts = np.array([box[0] for box in boxes])
        y_coords = top_lefts[:, 1]
        sorted_idx = np.argsort(y_coords)
        diffs = np.diff(y_coords[sorted_idx])
        split_points = np.where(diffs > threshold)[0] + 1
        return np.split(sorted_idx, split_points)

    @staticmethod
    def cluster_columns(boxes, threshold=15):
        """列聚类"""
        top_lefts = np.array([box[0] for box in boxes])
        x_coords = top_lefts[:, 0]
        sorted_idx = np.argsort(x_coords)
        diffs = np.diff(x_coords[sorted_idx])
        split_points = np.where(diffs > threshold)[0] + 1
        return np.split(sorted_idx, split_points)

    def reconstruct_table(self, rec_texts, dt_polys):
        """重建表格"""
        print("\n📊 表格重建中...")
        start_time = time.time()

        try:
            boxes = [self.parse_poly_str(p) for p in dt_polys]
            row_indices = self.cluster_rows(boxes)
            col_indices = self.cluster_columns(boxes)

            # 创建空表格
            max_rows = len(row_indices)
            max_cols = len(col_indices)
            table = [[""] * max_cols for _ in range(max_rows)]

            # 填充表格
            for i, row in enumerate(row_indices):
                for j, col in enumerate(col_indices):
                    cell_texts = []
                    for k in row:
                        if k in col:
                            cell_texts.append(rec_texts[k])
                    table[i][j] = " ".join(cell_texts)

            df = pd.DataFrame(table)

            print(f"✅ 表格重建完成，耗时: {time.time() - start_time:.2f}秒")
            return df
        except Exception as e:
            print(f"❌ 表格重建失败: {str(e)}")
            raise


def get_image_files(folder_path):
    """获取文件夹中所有图片文件"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(image_extensions):
            image_files.append(file)

    return image_files


def process_single_image(processor, image_path, base_name):
    """处理单张图片"""
    try:
        # OCR识别
        json_path, data = processor.run_structure_ocr(image_path)

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
        df = processor.reconstruct_table(rec_texts, dt_polys)

        # 保存结果到 ocr_results 文件夹 (保存为TXT和Excel文件)
        os.makedirs("ocr_results", exist_ok=True)

        # 保存为TXT文件
        txt_path = os.path.join("ocr_results", f"{base_name}_reconstructed_table.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write('\t'.join(df.columns.astype(str)) + '\n')
            # 写入数据行
            for _, row in df.iterrows():
                f.write('\t'.join(row.astype(str)) + '\n')

        # 保存为Excel文件
        excel_path = os.path.join("ocr_xlsx", f"{base_name}_reconstructed_table.xlsx")
        df.to_excel(excel_path, index=False)

        # 结果预览
        print(f"\n📄 {base_name} 表格预览:")
        print(df.head())
        print(f"\n✅ 结果已保存至: {txt_path} 和 {excel_path}")

        return True
    except Exception as e:
        print(f"\n❌ 处理 {image_path} 失败: {str(e)}")
        return False

def main():
    # 总计时
    total_start = time.time()
    print("\n" + "=" * 50)
    print("🛠️ 表格OCR处理程序启动")
    print("=" * 50)

    # 图片文件夹路径
    image_folder = "image"
    if not os.path.exists(image_folder):
        print(f"\n❌ 图片文件夹不存在: {image_folder}")
        return

    # 获取所有图片文件
    image_files = get_image_files(image_folder)
    if not image_files:
        print(f"\n❌ 在 {image_folder} 文件夹中未找到图片文件")
        return

    print(f"\n📁 找到 {len(image_files)} 个图片文件:")
    for img_file in image_files:
        print(f"  - {img_file}")

    try:
        # 初始化
        init_start = time.time()
        processor = TableOCRProcessor()
        print(f"\n🔄 初始化总耗时: {time.time() - init_start:.2f}秒")

        # 处理所有图片
        success_count = 0
        for img_file in image_files:
            image_path = os.path.join(image_folder, img_file)
            base_name = os.path.splitext(img_file)[0]

            print(f"\n{'-' * 50}")
            print(f"正在处理: {img_file}")
            print(f"{'-' * 50}")

            if process_single_image(processor, image_path, base_name):
                success_count += 1

        # 总结
        print(f"\n{'=' * 50}")
        print(f"🏁 批量处理完成！成功处理 {success_count}/{len(image_files)} 个文件")

    except Exception as e:
        print(f"\n❌ 批量处理过程中发生错误: {str(e)}")
        return

    # 总耗时
    total_time = time.time() - total_start
    print(f"⏱️  全部处理完成！总耗时: {total_time:.2f}秒")
    print("=" * 50)


if __name__ == "__main__":
    try:
        paddle.set_default_dtype('float16')
        print("\n🔼 已启用float16混合精度")
    except:
        print("\n🔽 当前环境不支持float16，使用默认精度")

    main()
