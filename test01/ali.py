import os
import paddle
from paddleocr import PPStructureV3
import time
import json
import psutil

class TableOCRProcessor:
    def __init__(self):
        self._print_system_info()
        self._init_paddle()
        self.model = self._init_model()

    def _print_system_info(self):
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
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
            print("\n✅ 已启用GPU加速")
        else:
            paddle.set_device('cpu')
            print("\n⚠️ 未检测到GPU，使用CPU模式")

    def _init_model(self):
        print("\n🔥 模型初始化中...")
        start_time = time.time()
        model = PPStructureV3()
        try:
            dummy_img = (255 * paddle.rand([100, 100, 3])).numpy().astype('uint8')
            model.predict(dummy_img)
        except Exception:
            pass
        print(f"模型初始化完成，耗时: {time.time() - start_time:.2f}秒")
        return model

    def process_single_image(self, image_path: str, output_dir: str = "output") -> dict:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        print(f"\n🖼️ 处理图片（不缩放）: {image_path}")

        start_time = time.time()
        result = self.model.predict(image_path)
        print(f"  OCR推理耗时: {time.time() - start_time:.2f}秒")

        json_path = os.path.join(output_dir, f"{base_name}_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"  结果保存到: {json_path}")

        return {"image": image_path, "json_path": json_path, "success": True}

def batch_process_images(image_paths, output_dir="output"):
    processor = TableOCRProcessor()
    results = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"❌ 文件不存在: {img_path}")
            continue
        result = processor.process_single_image(img_path, output_dir)
        results.append(result)
    print(f"\n全部处理完成，共处理{len(results)}张图片。")

if __name__ == "__main__":
    # 避免 sklearn 内存泄漏警告（Windows下常见）
    os.environ["OMP_NUM_THREADS"] = "1"

    # 设置float16混合精度（如果你确定支持）
    try:
        paddle.set_default_dtype('float16')
        print("\n🔼 已启用float16混合精度")
    except:
        print("\n🔽 当前环境不支持float16，使用默认精度")

    # 需要处理的图片列表，放入你的原图路径
    image_files = ["t.jpg", "wf.jpg"]
    batch_process_images(image_files)
