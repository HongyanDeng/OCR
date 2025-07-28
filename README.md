# OCR识别（打印体+手写体+表格）
#一、准备工作

##1. 配置python环境

采用Miniconda+PyCharm

###1.1 下载安装miniconda

官网链接：https://www.anaconda.com/download/success

点击下载好的.exe运行文件，选择安装路径，完成安装

验证是否安装成功

电脑中搜索并打开 Anaconda Prompt，输入命令：

```
conda --version
```

输出类似：

```
conda 24.1.2
```

表示安装成功



###1.2 下载安装PyCharm

官网链接：https://www.jetbrains.com/pycharm/download/?section=windows

点击下载好的.exe文件，选择安装路径，完成安装



###1.3 创建Python虚拟环境

打开Anaconda Prompt，输入命令：

```
conda create -n ocr python=3.9
```

其中：ocr为创建的虚拟环境名称，python版本为3.9

输入：

```
conda activate ocr
```

激活虚拟环境



###1.4 安装依赖包Paddlepaddle和PaddleOCR

激活虚拟环境ocr后，输入命令：

```
pip install paddlepaddle==2.5.0 -i https://mirror.baidu.com/pypi/simple
pip install paddleocr==3.1.0 -i https://mirror.baidu.com/pypi/simple
```

验证安装是否成功：

```
pip show paddlepaddle
pip show paddleocr
```

输出类似：

```
(m2) C:\Users\deng>pip show paddlepaddle
Name: paddlepaddle
Version: 2.5.0
Summary: Parallel Distributed Deep Learning
Home-page: UNKNOWN
Author:
Author-email: Paddle-better@baidu.com
License: Apache Software License
Location: c:\users\deng\.conda\envs\m2\lib\site-packages
Requires: astor, decorator, httpx, numpy, opt-einsum, paddle-bfloat, Pillow, protobuf
Required-by:

(m2) C:\Users\deng>pip show paddleocr
Name: paddleocr
Version: 3.1.0
Summary: Awesome multilingual OCR and document parsing toolkits based on PaddlePaddle
Home-page: https://github.com/PaddlePaddle/PaddleOCR
Author:
Author-email: PaddlePaddle <Paddle-better@baidu.com>
License: Apache License 2.0
Location: c:\users\deng\.conda\envs\m2\lib\site-packages
Requires: paddlex, PyYAML, typing-extensions
Required-by:
```



## 2.创建项目

打开Pycharm，创建新项目OCR

解释器类型选择**自定义环境**

环境类型选择**选择现有**

类型选择**Python**

Python路径选择虚拟环境路径，通常在C:\Users\deng\\.conda\envs\ocr\python.exe

选择完成后，点击创建



#二、OCR识别

创建文件ocr.py：

```python
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
        print("\n 系统信息:")
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
            print("\n 已启用GPU加速")
        else:
            paddle.set_device('cpu')
            print("\n 未检测到GPU，使用CPU模式")

    def _init_model(self):
        """初始化PPStructureV3模型"""
        print("\n 模型初始化中...")
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
        print("\n 图像预处理中...")
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
        print("\n OCR识别开始")
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

            print(f"\n OCR识别完成，总耗时: {time.time() - start_time:.2f}秒")
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
        print("\n 表格重建中...")
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

            print(f" 表格重建完成，耗时: {time.time() - start_time:.2f}秒")
            return df
        except Exception as e:
            print(f" 表格重建失败: {str(e)}")
            raise


def main():
    # 总计时
    total_start = time.time()
    print("\n" + "=" * 50)
    print(" 表格OCR处理程序启动")
    print("=" * 50)

    # 输入图像
    image_path = "wf2.jpg"
    if not os.path.exists(image_path):
        print(f"\n❌ 图片文件不存在: {image_path}")
        return

    try:
        # 初始化
        init_start = time.time()
        processor = TableOCRProcessor()
        print(f"\n 初始化总耗时: {time.time() - init_start:.2f}秒")

        # OCR识别
        ocr_start = time.time()
        json_path = processor.run_structure_ocr(image_path)
        print(f"\n OCR总耗时: {time.time() - ocr_start:.2f}秒")

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
        print(f"\n 表格重建总耗时: {time.time() - rebuild_start:.2f}秒")

        # 保存结果
        csv_path = "simple_reconstructed_table.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 结果预览
        print("\n 表格预览:")
        print(df.head())
        print(f"\n 结果已保存至: {csv_path}")

    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        return

    # 总耗时
    total_time = time.time() - total_start
    print("\n" + "=" * 50)
    print(f" 全部处理完成！总耗时: {total_time:.2f}秒")
    print("=" * 50)


if __name__ == "__main__":
    try:
        paddle.set_default_dtype('float16')
        print("\n 已启用float16混合精度")
    except:
        print("\n 当前环境不支持float16，使用默认精度")

    main()
```

#三、连接Milvus

创建connect.py文件

```
# upload_to_milvus.py
import pandas as pd
import numpy as np
import os
from pymilvus import (
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    Collection
)
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType


class MilvusUploader:
    def __init__(self, milvus_host="192.168.31.132", milvus_port="19530"):
        """
        初始化Milvus连接

        Args:
            milvus_host (str): Milvus服务器地址
            milvus_port (str): Milvus服务器端口
        """
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self._connect_to_milvus()

    def _connect_to_milvus(self):
        """连接到Milvus服务器"""
        try:
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            print(f"✅ 成功连接到Milvus服务器 {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            print(f"❌ 连接Milvus失败: {str(e)}")
            raise

    def create_schema(self):
        """
        创建表格数据的Milvus Schema

        Returns:
            CollectionSchema: Milvus集合模式
        """
        # 定义字段 - 使用最简单的数据类型
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="row_index", dtype=DataType.INT64, description="行索引"),
            FieldSchema(name="col_index", dtype=DataType.INT64, description="列索引"),
            FieldSchema(name="col_name", dtype=DataType.VARCHAR, max_length=256, description="列名"),
            FieldSchema(name="cell_content", dtype=DataType.VARCHAR, max_length=65535, description="单元格内容"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="内容向量表示")  # 进一步减小向量维度
        ]

        schema = CollectionSchema(
            fields=fields,
            description="OCR识别的表格数据"
        )
        return schema

    def create_or_get_collection(self, collection_name):
        """
        创建或获取Milvus集合

        Args:
            collection_name (str): 集合名称

        Returns:
            Collection: Milvus集合对象
        """
        try:
            schema = self.create_schema()

            # 如果集合已存在，删除它并重新创建
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                print(f"🗑️ 已删除现有集合: {collection_name}")

            # 创建新集合
            collection = Collection(name=collection_name, schema=schema)
            print(f"🆕 创建新集合: {collection_name}")

            # 创建索引
            index_params = {
                "index_type": "FLAT",
                "metric_type": "L2",
                "params": {}
            }

            collection.create_index(field_name="embedding", index_params=index_params)
            print(f".CreateIndex: 在embedding字段上创建索引")

            return collection

        except Exception as e:
            print(f"❌ 集合操作失败: {str(e)}")
            raise

    def text_to_vector(self, text, dim=128):
        """
        将文本转换为向量表示（简单实现）

        Args:
            text (str): 输入文本
            dim (int): 向量维度

        Returns:
            list: 向量表示
        """
        # 最简单可靠的向量生成方法
        if not text or not str(text).strip():
            return [0.1] * dim

        # 使用hash值生成固定向量
        hash_val = hash(str(text))
        np.random.seed(abs(hash_val) % (2 ** 32))
        vector = np.random.random(dim).tolist()

        # 归一化
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector

    def create_bulk_writer(self, collection_name, s3_config=None):
        """
        创建RemoteBulkWriter用于批量上传数据

        Args:
            collection_name (str): 集合名称
            s3_config (dict): S3配置参数

        Returns:
            RemoteBulkWriter: 批量写入器
        """
        try:
            # 默认S3配置 - 根据您的服务器IP配置
            if s3_config is None:
                s3_config = {
                    "endpoint": "192.168.31.132:9000",  # MinIO服务地址
                    "access_key": "minioadmin",  # MinIO访问密钥
                    "secret_key": "minioadmin",  # MinIO秘密密钥
                    "bucket_name": "a-bucket",  # 存储桶名称（需要先创建）
                    "secure": False  # 本地部署，不使用HTTPS
                }

            # 创建集合Schema
            schema = self.create_schema()

            # S3连接参数
            conn = RemoteBulkWriter.S3ConnectParam(
                endpoint=s3_config["endpoint"],
                access_key=s3_config["access_key"],
                secret_key=s3_config["secret_key"],
                bucket_name=s3_config["bucket_name"],
                secure=s3_config["secure"]
            )

            # 创建BulkWriter
            writer = RemoteBulkWriter(
                schema=schema,
                remote_path=f"/{collection_name}/",
                connect_param=conn,
                file_type=BulkFileType.PARQUET
            )

            print(f"✅ BulkWriter创建成功，目标路径: /{collection_name}/")
            return writer

        except Exception as e:
            print(f"❌ 创建BulkWriter失败: {str(e)}")
            raise

    def load_csv_data(self, csv_path):
        """
        加载CSV文件数据

        Args:
            csv_path (str): CSV文件路径

        Returns:
            pandas.DataFrame: 加载的数据
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

            # 使用最简单的参数读取CSV
            df = pd.read_csv(csv_path, encoding="utf-8", dtype=object, keep_default_na=False, na_values=[])
            print(f"📄 成功读取CSV文件: {csv_path}")
            print(f"📊 数据形状: {df.shape[0]}行 x {df.shape[1]}列")

            return df

        except Exception as e:
            print(f"❌ 读取CSV文件失败: {str(e)}")
            raise

    def prepare_data_for_upload(self, df, vector_dim=128):
        """
        准备上传到Milvus的数据

        Args:
            df (pandas.DataFrame): 表格数据
            vector_dim (int): 向量维度

        Returns:
            dict: 准备好的数据字典
        """
        try:
            row_indices = []
            col_indices = []
            col_names = []
            cell_contents = []
            embeddings = []

            print(f"🔄 开始处理数据，共{len(df)}行，{len(df.columns)}列")

            # 遍历每一行数据
            for row_idx in range(len(df)):
                row = df.iloc[row_idx]
                # 遍历每一列
                for col_idx in range(len(df.columns)):
                    col_name = df.columns[col_idx]
                    cell_value = row[col_idx]

                    # 确保值是字符串
                    cell_value_str = str(cell_value) if cell_value is not None and str(cell_value) != 'nan' else ""
                    col_name_str = str(col_name) if col_name is not None else f"col_{col_idx}"

                    # 添加数据 - 确保使用Python原生int类型
                    row_indices.append(int(row_idx))
                    col_indices.append(int(col_idx))
                    col_names.append(col_name_str[:255])  # 限制长度
                    cell_contents.append(cell_value_str)
                    embeddings.append(self.text_to_vector(cell_value_str, vector_dim))

                    # 调试信息（仅显示前几条）
                    if len(row_indices) <= 3:
                        print(
                            f"📝 记录 {len(row_indices)}: 行={row_idx}, 列={col_idx}, 列名={col_name_str}, 内容='{cell_value_str[:30]}...'")

            data = {
                "row_index": [int(x) for x in row_indices],  # 确保是Python int
                "col_index": [int(x) for x in col_indices],  # 确保是Python int
                "col_name": col_names,
                "cell_content": cell_contents,
                "embedding": embeddings
            }

            print(f"✅ 准备好{len(row_indices)}条记录用于上传")
            return data

        except Exception as e:
            print(f"❌ 准备上传数据失败: {str(e)}")
            raise

    def upload_csv_to_milvus(self, csv_path, collection_name, s3_config=None, batch_size=5):
        """
        将CSV文件上传到Milvus

        Args:
            csv_path (str): CSV文件路径
            collection_name (str): Milvus集合名称
            s3_config (dict): S3配置参数
            batch_size (int): 批处理大小
        """
        try:
            print(f"\n📤 开始上传CSV文件到Milvus...")
            print(f"📁 CSV文件: {csv_path}")
            print(f"📦 集合名称: {collection_name}")

            # 创建BulkWriter
            writer = self.create_bulk_writer(collection_name, s3_config)

            # 加载数据
            df = self.load_csv_data(csv_path)

            # 准备数据
            data = self.prepare_data_for_upload(df, vector_dim=128)  # 使用更小的向量维度

            # 分批写入数据
            total_records = len(data["row_index"])
            uploaded_records = 0
            successful_batches = 0

            for i in range(0, total_records, batch_size):
                batch_end = min(i + batch_size, total_records)



                # 构建批次数据
                batch_data = {}
                for key in ["row_index", "col_index", "col_name", "cell_content", "embedding"]:
                    batch_data[key] = data[key][i:batch_end]

                # 确保数据类型完全正确并打印调试信息
                final_batch_data = {}
                print(f"\n🔍 验证批次 {i // batch_size + 1} 数据:")


                for key, value_list in batch_data.items():
                    if key in ["row_index", "col_index"]:
                        processed_list = []
                        for x in value_list:
                            try:
                                processed_list.append(int(float(str(x))))
                            except:
                                processed_list.append(np.int64(0))
                        final_batch_data[key] = processed_list
                        print(f"  {key}: {processed_list[:5]}")
                        print(f"  {key} types: {[type(x) for x in processed_list[:5]]}")  # 打印类型
                    elif key in ["col_name", "cell_content"]:
                        # 确保是字符串列表
                        processed_list = [str(x) if x is not None else "" for x in value_list]
                        final_batch_data[key] = processed_list
                        print(f"  {key}: {[x[:20] for x in processed_list[:5]]}")
                    elif key == "embedding":
                        # 确保是浮点数向量列表
                        validated_embeddings = []
                        for emb in value_list:
                            if isinstance(emb, (list, tuple)) and len(emb) == 128 :
                                validated_embeddings.append([float(x) for x in emb])
                            else:
                                # 使用默认向量
                                validated_embeddings.append([0.1] * 128)
                        final_batch_data[key] = validated_embeddings
                        print(f"  {key}: 长度={[len(x) for x in validated_embeddings[:3]]}")
                    else:
                        final_batch_data[key] = list(value_list)

                # 打印整个 final_batch_data 的结构
                print(f"  final_batch_data: {final_batch_data}")
                # ===== 添加的类型检查代码 =====
                print(f"数据类型检查 - row_index: {[type(x) for x in final_batch_data['row_index'][:5]]}")
                print(f"数据类型检查 - col_index: {[type(x) for x in final_batch_data['col_index'][:5]]}")

                try:
                    writer.append_row(final_batch_data)
                    batch_record_count = len(final_batch_data["row_index"])
                    uploaded_records += batch_record_count
                    successful_batches += 1
                    print(
                        f"✅ 成功处理批次 {i // batch_size + 1}: {batch_record_count} 条记录 (累计: {uploaded_records}/{total_records})")
                except Exception as e:
                    print(f"❌ 批次写入失败 (记录 {i + 1}-{batch_end}): {str(e)}")

            # 提交数据
            try:
                writer.commit()
                print(f"✅ 成功提交数据，共上传{uploaded_records}条记录到Milvus集合 '{collection_name}'")
                print(f"   成功批次数量: {successful_batches}")
            except Exception as e:
                print(f"⚠️  提交数据时出错: {str(e)}")

            # 加载集合
            collection = self.create_or_get_collection(collection_name)
            collection.load()
            print(f"🔄 集合 '{collection_name}' 已加载到内存")

        except Exception as e:
            print(f"❌ 上传数据到Milvus失败: {str(e)}")
            raise

    def query_collection_info(self, collection_name):
        """
        查询集合信息

        Args:
            collection_name (str): 集合名称
        """
        try:
            if utility.has_collection(collection_name):
                collection = Collection(name=collection_name)
                print(f"\n🔍 集合 '{collection_name}' 信息:")
                print(f"   行数: {collection.num_entities}")
                print(f"   字段: {[field.name for field in collection.schema.fields]}")

                # 显示部分数据
                if collection.num_entities > 0:
                    try:
                        collection.load()
                        results = collection.query(expr="row_index < 3",
                                                   output_fields=["row_index", "col_index", "col_name", "cell_content"],
                                                   limit=10)
                        print(f"   示例数据:")
                        for i, result in enumerate(results):
                            print(
                                f"     {i + 1}. 行:{result.get('row_index', 'N/A')}, 列:{result.get('col_index', 'N/A')}, "
                                f"列名:'{result.get('col_name', 'N/A')[:20]}...', 内容:'{result.get('cell_content', 'N/A')[:30]}...'")
                    except Exception as e:
                        print(f"   无法查询示例数据: {str(e)}")
            else:
                print(f"⚠️  集合 '{collection_name}' 不存在")
        except Exception as e:
            print(f"❌ 查询集合信息失败: {str(e)}")


def main():
    """主函数 - 上传CSV到Milvus"""
    print("=" * 60)
    print("📦 CSV文件上传到Milvus工具")
    print("=" * 60)

    # 配置参数
    CSV_FILE_PATH = "simple_reconstructed_table.csv"  # CSV文件路径
    COLLECTION_NAME = "ocr_table_data"  # Milvus集合名称

    # Milvus服务器配置
    MILVUS_HOST = "192.168.31.132"  # Milvus服务器IP地址
    MILVUS_PORT = "19530"  # Milvus服务端口

    # S3配置（MinIO配置）
    S3_CONFIG = {
        "endpoint": "192.168.31.132:9000",  # MinIO服务地址
        "access_key": "minioadmin",  # MinIO访问密钥
        "secret_key": "minioadmin",  # MinIO秘密密钥
        "bucket_name": "a-bucket",  # 需要先在MinIO中创建此存储桶
        "secure": False  # 本地部署，不使用HTTPS
    }

    try:
        # 创建上传器实例
        uploader = MilvusUploader(milvus_host=MILVUS_HOST, milvus_port=MILVUS_PORT)

        # 上传CSV到Milvus
        uploader.upload_csv_to_milvus(
            csv_path=CSV_FILE_PATH,
            collection_name=COLLECTION_NAME,
            s3_config=S3_CONFIG,
            batch_size=5  # 使用最小的批次大小
        )

        # 显示集合信息
        uploader.query_collection_info(COLLECTION_NAME)

        print("\n✅ 数据上传完成!")

    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
        return


if __name__ == "__main__":
    main()
```