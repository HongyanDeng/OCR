# upload_to_milvus.py
import os
import json
import time
import pandas as pd
import numpy as np
from pymilvus import (
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    Collection
)
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType, bulk_import, list_import_jobs

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
        self.milvus_url = f"http://{milvus_host}:{milvus_port}"  # HTTP接口调用地址

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
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="row_index", dtype=DataType.INT64, description="行索引"),
            FieldSchema(name="col_index", dtype=DataType.INT64, description="列索引"),
            FieldSchema(name="col_name", dtype=DataType.VARCHAR, max_length=256, description="列名"),
            FieldSchema(name="cell_content", dtype=DataType.VARCHAR, max_length=65535, description="单元格内容"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="内容向量表示")
        ]
        schema = CollectionSchema(fields=fields, description="OCR识别的表格数据")
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
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                print(f"🗑️ 已删除现有集合: {collection_name}")

            collection = Collection(name=collection_name, schema=schema)
            print(f"🆕 创建新集合: {collection_name}")

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
        简单文本转向量（归一化随机向量）

        Args:
            text (str): 输入文本
            dim (int): 向量维度

        Returns:
            list: 向量表示
        """
        if not text or not str(text).strip():
            return [0.1] * dim
        hash_val = hash(str(text))
        np.random.seed(abs(hash_val) % (2**32))
        vector = np.random.random(dim).tolist()
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
            if s3_config is None:
                s3_config = {
                    "endpoint": "192.168.31.132:9000",
                    "access_key": "minioadmin",
                    "secret_key": "minioadmin",
                    "bucket_name": "a-bucket",
                    "secure": False
                }

            schema = self.create_schema()

            conn = RemoteBulkWriter.S3ConnectParam(
                endpoint=s3_config["endpoint"],
                access_key=s3_config["access_key"],
                secret_key=s3_config["secret_key"],
                bucket_name=s3_config["bucket_name"],
                secure=s3_config["secure"]
            )

            writer = RemoteBulkWriter(
                schema=schema,
                remote_path=f"milvus_upload_tmp/{collection_name}".replace("\\", "/"),
                connect_param=conn,
                file_type=BulkFileType.PARQUET
            )

            print(f"✅ BulkWriter创建成功，目标路径: milvus_upload_tmp/{collection_name}")
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

            for row_idx in range(len(df)):
                row = df.iloc[row_idx]
                for col_idx in range(len(df.columns)):
                    col_name = df.columns[col_idx]
                    cell_value = row.iloc[col_idx]
                    cell_value_str = str(cell_value) if cell_value is not None and str(cell_value) != 'nan' else ""
                    col_name_str = str(col_name) if col_name is not None else f"col_{col_idx}"

                    row_indices.append(int(row_idx))
                    col_indices.append(int(col_idx))
                    col_names.append(col_name_str[:255])
                    cell_contents.append(cell_value_str)
                    embeddings.append(self.text_to_vector(cell_value_str, vector_dim))

                    if len(row_indices) <= 3:
                        print(
                            f"📝 记录 {len(row_indices)}: 行={row_idx}, 列={col_idx}, 列名={col_name_str}, 内容='{cell_value_str[:30]}...'")

            data = {
                "row_index": [int(x) for x in row_indices],
                "col_index": [int(x) for x in col_indices],
                "col_name": col_names,
                "cell_content": cell_contents,
                "embedding": embeddings
            }
            print(f"✅ 准备好{len(row_indices)}条记录用于上传")
            return data

        except Exception as e:
            print(f"❌ 准备上传数据失败: {str(e)}")
            raise

    def upload_csv_to_milvus(self, csv_path, collection_name, s3_config=None, batch_size=1000):
        """
        将CSV文件上传到Milvus并触发bulk_import导入

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

            writer = self.create_bulk_writer(collection_name, s3_config)
            df = self.load_csv_data(csv_path)
            data = self.prepare_data_for_upload(df, vector_dim=128)

            total_records = len(data["row_index"])
            uploaded_records = 0
            successful_batches = 0

            for i in range(0, total_records, batch_size):
                batch_end = min(i + batch_size, total_records)

                batch_data = {}
                for key in ["row_index", "col_index", "col_name", "cell_content", "embedding"]:
                    batch_data[key] = data[key][i:batch_end]

                final_batch_data = {}

                print(f"\n🔍 验证批次 {i // batch_size + 1} 数据:")
                for key, value_list in batch_data.items():
                    if key in ["row_index", "col_index"]:
                        processed_list = []
                        for x in value_list:
                            try:
                                processed_list.append(int(float(str(x))))
                            except Exception as e:
                                print(f"⚠️ 转换 {key} 失败，值: {x}，错误: {e}")
                                processed_list.append(int(0))
                        final_batch_data[key] = processed_list
                        print(f"数据类型检查 - {key}: {[type(x) for x in final_batch_data[key][:5]]}")
                        print(f"  {key}: {processed_list[:5]}")
                    elif key in ["col_name", "cell_content"]:
                        processed_list = [str(x) if x is not None else "" for x in value_list]
                        final_batch_data[key] = processed_list
                        print(f"  {key}: {[x[:20] for x in processed_list[:5]]}")
                    elif key == "embedding":
                        validated_embeddings = []
                        for emb in value_list:
                            if isinstance(emb, (list, tuple)) and len(emb) == 128:
                                validated_embeddings.append([float(x) for x in emb])
                            else:
                                validated_embeddings.append([0.1] * 128)
                        final_batch_data[key] = validated_embeddings
                        print(f"  {key}: 长度={[len(x) for x in validated_embeddings[:3]]}")
                    else:
                        final_batch_data[key] = list(value_list)

                # 写入批次数据
                writer.append_row(final_batch_data)
                batch_record_count = len(final_batch_data["row_index"])
                uploaded_records += batch_record_count
                successful_batches += 1
                print(f"✅ 成功处理批次 {i // batch_size + 1}: {batch_record_count} 条记录 (累计: {uploaded_records}/{total_records})")

            writer.commit()
            print(f"✅ BulkWriter提交完成，生成文件数: {len(writer.batch_files)}")

            # 调用bulk_import异步导入Milvus
            parquet_files = [[f] for f in writer.batch_files]  # bulk_import需要二维列表
            job_id = self.trigger_bulk_import(collection_name, parquet_files)
            print(f"📥 导入任务已提交，Job ID: {job_id}")

            # 创建并加载集合，方便后续查询
            collection = self.create_or_get_collection(collection_name)
            collection.load()
            print(f"🔄 集合 '{collection_name}' 已加载到内存")

            return job_id

        except Exception as e:
            print(f"❌ 上传数据到Milvus失败: {str(e)}")
            raise

    def trigger_bulk_import(self, collection_name, parquet_files):
        """
        调用bulk_import接口触发异步导入

        Args:
            collection_name (str): 集合名称
            parquet_files (list[list[str]]): parquet文件路径二维列表

        Returns:
            str: bulk_import任务ID
        """
        try:
            resp = bulk_import(
                url=self.milvus_url,
                collection_name=collection_name,
                files=parquet_files
            )
            job_id = resp.json()['data']['jobId']
            return job_id
        except Exception as e:
            print(f"❌ bulk_import调用失败: {str(e)}")
            raise

    def list_import_jobs_status(self, collection_name):
        """
        查询bulk_import异步导入任务状态

        Args:
            collection_name (str): 集合名称
        """
        try:
            resp = list_import_jobs(url=self.milvus_url, collection_name=collection_name)
            print(json.dumps(resp.json(), indent=4))
        except Exception as e:
            print(f"❌ 查询导入任务失败: {str(e)}")

    def query_collection_info(self, collection_name):
        """
        查询集合信息及示例数据

        Args:
            collection_name (str): 集合名称
        """
        try:
            if utility.has_collection(collection_name):
                collection = Collection(name=collection_name)
                print(f"\n🔍 集合 '{collection_name}' 信息:")
                print(f"   行数: {collection.num_entities}")
                print(f"   字段: {[field.name for field in collection.schema.fields]}")

                if collection.num_entities > 0:
                    try:
                        collection.load()
                        results = collection.query(
                            expr="row_index < 3",
                            output_fields=["row_index", "col_index", "col_name", "cell_content"],
                            limit=10
                        )
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
    print("=" * 60)
    print("📦 CSV文件上传到Milvus工具")
    print("=" * 60)

    CSV_FILE_PATH = "simple_reconstructed_table.csv"
    COLLECTION_NAME = "ocr_table_data"

    MILVUS_HOST = "192.168.31.132"
    MILVUS_PORT = "19530"

    S3_CONFIG = {
        "endpoint": "192.168.31.132:9000",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
        "bucket_name": "a-bucket",
        "secure": False
    }

    try:
        uploader = MilvusUploader(milvus_host=MILVUS_HOST, milvus_port=MILVUS_PORT)

        # 上传CSV，触发bulk_import，返回任务ID
        job_id = uploader.upload_csv_to_milvus(
            csv_path=CSV_FILE_PATH,
            collection_name=COLLECTION_NAME,
            s3_config=S3_CONFIG,
            batch_size=1000  # 根据需要调整批次大小
        )

        print(f"\n✅ 数据上传并导入任务提交完成! Job ID: {job_id}\n")

        # 查询集合信息和示例
        uploader.query_collection_info(COLLECTION_NAME)

        # 查询导入任务状态（示例）
        print("\n📄 查询导入任务状态:")
        uploader.list_import_jobs_status(COLLECTION_NAME)

    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")

if __name__ == "__main__":
    main()
