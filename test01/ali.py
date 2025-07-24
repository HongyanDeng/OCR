import os
import paddle
from paddleocr import PPStructureV3
from PIL import Image
import json
import pandas as pd
import numpy as np
import time
import psutil
from typing import List


class TableOCRProcessor:
    def __init__(self):
        """初始化OCR处理器（单例模式）"""
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
            paddle.set_flags({
                'FLAGS_conv_workspace_size_limit': 128,
                'FLAGS_cudnn_exhaustive_search': 1,
                'FLAGS_allocator_strategy': 'auto_growth'
            })
        else:
            paddle.set_device('cpu')
            print("\n⚠️ 未检测到GPU，使用CPU模式")

    def _init_model(self):
        """初始化PPStructureV3模型"""
        print("\n🔥 模型初始化中...")
        start_time = time.time()

        model = PPStructureV3()  # 不传递参数避免兼容性问题

        # 小型预热
        try:
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            model.predict(dummy_img)
        except Exception as e:
            print(f"预热警告: {str(e)}")

        print(f"模型初始化完成，耗时: {time.time() - start_time:.2f}秒")
        return model

    def preprocess_image(self, image_path: str, max_size: int = 1200) -> Image.Image:
        """
        图像预处理（自动处理RGBA格式和缩放）
        :param image_path: 图片路径
        :param max_size: 最大边长像素
        :return: PIL.Image对象
        """
        print(f"\n🖼️ 预处理图片: {os.path.basename(image_path)}")
        try:
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                print("  已转换RGBA为RGB格式")

            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                print(f"  图片从 ({w},{h}) 缩放至 ({new_w},{new_h})")

            return img
        except Exception as e:
            print(f"❌ 图片预处理失败: {str(e)}")
            raise

    def process_single_image(self, image_path: str, output_dir: str = "output") -> dict:
        """
        处理单张图片并保存结果
        :param image_path: 图片路径
        :param output_dir: 输出目录
        :return: 包含结果路径的字典
        """
        result_info = {
            "image": os.path.basename(image_path),
            "json_path": None,
            "csv_path": None,
            "success": False
        }

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        try:
            # 1. 预处理图像
            img = self.preprocess_image(image_path)
            temp_img_path = os.path.join(output_dir, f"temp_{base_name}.jpg")
            img.save(temp_img_path, quality=95, subsampling=0)

            # 2. 执行OCR
            print(f"🔍 正在识别: {base_name}")
            ocr_start = time.time()
            result = self.model.predict(temp_img_path)
            print(f"  OCR核心耗时: {time.time() - ocr_start:.2f}秒")

            # 3. 保存JSON结果
            json_path = os.path.join(output_dir, f"{base_name}_result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            result_info["json_path"] = json_path

            # 4. 提取表格数据
            if result and isinstance(result, list):
                overall_ocr_res = result[0].get('overall_ocr_res', {})
                rec_texts = overall_ocr_res.get('rec_texts', [])
                dt_polys = overall_ocr_res.get('dt_polys', [])

                if len(rec_texts) == len(dt_polys):
                    df = self._reconstruct_table(rec_texts, dt_polys)
                    csv_path = os.path.join(output_dir, f"{base_name}_table.csv")
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    result_info["csv_path"] = csv_path
                    print(f"📊 表格已保存: {csv_path}")

            result_info["success"] = True
            return result_info

        except Exception as e:
            print(f"❌ 处理失败: {base_name} - {str(e)}")
            return result_info
        finally:
            # 清理临时文件
            if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

    def _reconstruct_table(self, rec_texts: List[str], dt_polys: List[str]) -> pd.DataFrame:
        """重建表格数据结构"""
        boxes = [self._parse_poly_str(p) for p in dt_polys]
        rows_indices = self._cluster_rows(boxes)
        top_lefts = np.array([box[0] for box in boxes])

        rows = []
        for row in rows_indices:
            row_sorted = sorted(row, key=lambda i: top_lefts[i, 0])
            rows.append([rec_texts[i] for i in row_sorted])

        max_cols = max(len(row) for row in rows)
        return pd.DataFrame([row + [''] * (max_cols - len(row)) for row in rows])

    @staticmethod
    def _parse_poly_str(poly_str: str) -> List[List[int]]:
        """解析坐标字符串"""
        cleaned = ''.join(c if c.isdigit() or c.isspace() else ' ' for c in poly_str)
        points = np.fromstring(cleaned, sep=' ', dtype=int).reshape(-1, 2)
        return points.tolist()

    @staticmethod
    def _cluster_rows(boxes: List[List[List[int]]], threshold: int = 15) -> List[np.ndarray]:
        """行聚类算法"""
        top_lefts = np.array([box[0] for box in boxes])
        y_coords = top_lefts[:, 1]
        sorted_idx = np.argsort(y_coords)
        diffs = np.diff(y_coords[sorted_idx])
        split_points = np.where(diffs > threshold)[0] + 1
        return np.split(sorted_idx, split_points)


def batch_process_images(image_paths: List[str], output_dir: str = "output"):
    """
    批量处理图片
    :param image_paths: 图片路径列表
    :param output_dir: 输出目录
    """
    print("\n" + "=" * 50)
    print(f"🛠️ 开始批量处理 {len(image_paths)} 张图片")
    print("=" * 50)

    total_start = time.time()
    processor = TableOCRProcessor()

    results = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"\n❌ 文件不存在: {img_path}")
            continue

        print("\n" + "-" * 50)
        print(f"📌 正在处理: {os.path.basename(img_path)}")
        print("-" * 50)

        start_time = time.time()
        result = processor.process_single_image(img_path, output_dir)
        result["process_time"] = time.time() - start_time
        results.append(result)

        if result["success"]:
            print(f"✅ 处理成功 (耗时: {result['process_time']:.2f}秒)")
        else:
            print(f"❌ 处理失败 (耗时: {result['process_time']:.2f}秒)")

    # 打印汇总信息
    print("\n" + "=" * 50)
    print("📊 批量处理结果汇总")
    print("=" * 50)
    success_count = sum(1 for r in results if r["success"])
    avg_time = sum(r["process_time"] for r in results) / len(results) if results else 0

    print(f"成功: {success_count}/{len(image_paths)}")
    print(f"平均耗时: {avg_time:.2f}秒/张")
    print(f"总耗时: {time.time() - total_start:.2f}秒")

    # 保存汇总日志
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": len(image_paths),
            "success_count": success_count,
            "average_time": avg_time,
            "details": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📝 详细日志已保存至: {summary_path}")


if __name__ == "__main__":
    # 设置混合精度
    try:
        paddle.set_default_dtype('float16')
        print("\n🔼 已启用float16混合精度")
    except:
        print("\n🔽 当前环境不支持float16，使用默认精度")

    # 要处理的图片列表
    image_files = ["t.jpg", "wf.jpg"]  # 添加更多图片路径

    # 过滤出实际存在的文件
    valid_images = [img for img in image_files if os.path.exists(img)]
    if not valid_images:
        print("\n❌ 没有找到可处理的图片文件")
    else:
        batch_process_images(valid_images, output_dir="ocr_results")