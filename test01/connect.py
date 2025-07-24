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
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="row_index", dtype=DataType.INT64, description="行索引"),
            FieldSchema(name="col_index", dtype=DataType.INT64, description="列索引"),
            FieldSchema(name="col_name", dtype=DataType.VARCHAR, max_length=256, description="列名"),
            FieldSchema(name="cell_content", dtype=DataType.VARCHAR, max_length=65535, description="单元格内容"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="内容向量表示")
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

            # 如果集合已存在，加载它
            if utility.has_collection(collection_name):
                collection = Collection(name=collection_name)
                print(f"📥 已加载现有集合: {collection_name}")
            else:
                # 创建新集合
                collection = Collection(name=collection_name, schema=schema)
                print(f"🆕 创建新集合: {collection_name}")

                # 创建索引
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 128}
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
        if not text or not str(text).strip():
            return [0.0] * dim

        # 使用简单哈希方法生成向量（仅作示例）
        np.random.seed(hash(str(text)) % (2 ** 32))
        vector = np.random.random(dim).tolist()
        # 归一化向量
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [x / norm for x in vector]
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

            df = pd.read_csv(csv_path, encoding="utf-8")
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

            # 遍历每一行数据
            for row_idx, row in df.iterrows():
                # 遍历每一列
                for col_idx, (col_name, cell_value) in enumerate(row.items()):
                    cell_value_str = str(cell_value) if not pd.isna(cell_value) else ""

                    row_indices.append(row_idx)
                    col_indices.append(col_idx)
                    col_names.append(str(col_name)[:255])  # 限制列名长度
                    cell_contents.append(cell_value_str)
                    embeddings.append(self.text_to_vector(cell_value_str, vector_dim))

            data = {
                "row_index": row_indices,
                "col_index": col_indices,
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
            data = self.prepare_data_for_upload(df)

            # 分批写入数据
            total_records = len(data["row_index"])
            uploaded_records = 0

            for i in range(0, total_records, batch_size):
                batch_end = min(i + batch_size, total_records)

                batch_data = {}
                for key, value_list in data.items():
                    batch_data[key] = value_list[i:batch_end]

                writer.append_row(batch_data)
                uploaded_records += len(batch_data["row_index"])
                print(f"📈 已处理: {uploaded_records}/{total_records} 条记录")

            # 提交数据
            writer.commit()
            print(f"✅ 成功上传{uploaded_records}条记录到Milvus集合 '{collection_name}'")

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
            batch_size=1000
        )

        # 显示集合信息
        uploader.query_collection_info(COLLECTION_NAME)

        print("\n✅ 数据上传完成!")

    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
        return


if __name__ == "__main__":
    main()
