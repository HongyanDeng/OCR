import os
import sys
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import json
import re  # 添加此行以修复正则表达式未定义的问题

import traceback
from ali import TableOCRProcessor  # 导入您已有的类

# 初始化 Flask 应用
app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 全局 OCR 处理器实例（初始化时创建一次）
processor = None


def load_model():
    """加载模型"""
    global processor
    if processor is None:  # 避免重复加载
        try:
            print("正在加载 OCR 模型...")
            processor = TableOCRProcessor()
            print("OCR 模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            traceback.print_exc()
            raise


# 在第一次请求时加载模型
@app.before_request
def before_first_request():
    global processor
    if processor is None:
        load_model()


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "model_loaded": processor is not None
    })


@app.route('/ocr/process', methods=['POST'])
def process_image():
    """处理 OCR 请求"""
    global processor

    # 确保模型已加载
    if processor is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({
                "error": f"模型加载失败: {str(e)}",
                "code": "MODEL_LOADING_ERROR"
            }), 500

    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            return jsonify({
                "error": "没有上传图片文件",
                "code": "MISSING_FILE"
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "error": "文件名为空",
                "code": "EMPTY_FILENAME"
            }), 400

        # 保存上传的文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 处理图片
        base_name = os.path.splitext(filename)[0]

        # OCR识别
        json_path, data = processor.run_structure_ocr(file_path)

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

        # 对OCR结果进行后处理（错别字纠正）
        processed_texts = processor.post_process_texts(rec_texts)

        # 质量评估
        quality_report = assess_ocr_quality(processed_texts)

        # 错别字和语义检查
        typo_issues = processor.detect_typos_and_inconsistencies(processed_texts)

        # 表格重建
        df = processor.reconstruct_table(processed_texts, dt_polys)

        # 保存处理结果
        result_data = {
            "filename": filename,
            "texts": processed_texts,
            "coordinates": [processor.parse_poly_str(p) for p in dt_polys],
            "quality_report": quality_report,
            "typo_issues": typo_issues,
            "table_data": df.to_dict('records') if not df.empty else [],
            "table_columns": df.columns.tolist() if not df.empty else []
        }

        # 保存为 JSON 文件
        result_file = os.path.join(PROCESSED_FOLDER, f"{base_name}_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)

        # 清理上传的文件
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            "success": True,
            "data": result_data,
            "result_file": result_file
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "code": "PROCESSING_ERROR"
        }), 500


@app.route('/ocr/result/<filename>', methods=['GET'])
def get_result(filename):
    """获取处理结果文件"""
    try:
        result_file = os.path.join(PROCESSED_FOLDER, filename)
        if not os.path.exists(result_file):
            return jsonify({
                "error": "结果文件不存在",
                "code": "FILE_NOT_FOUND"
            }), 404

        return send_file(result_file, as_attachment=True)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "code": "FILE_ACCESS_ERROR"
        }), 500


def assess_ocr_quality(rec_texts):
    """
    基于规则评估OCR质量（从原代码中复制）
    """
    if not rec_texts:
        return {"quality_score": 0, "issues": ["无识别结果"]}

    issues = []
    scores = []

    # 1. 检查空文本比例
    empty_count = sum(1 for text in rec_texts if not text.strip())
    empty_ratio = empty_count / len(rec_texts)
    if empty_ratio > 0.1:
        issues.append(f"空文本比例过高: {empty_ratio:.2%}")
    scores.append(max(0, 1 - empty_ratio))

    # 2. 检查文本长度合理性
    lengths = [len(text) for text in rec_texts if text.strip()]
    if lengths:
        avg_length = sum(lengths) / len(lengths)
        if avg_length < 1 or avg_length > 50:
            issues.append(f"平均文本长度异常: {avg_length:.1f}")
        else:
            scores.append(1.0)

    # 3. 检查特殊字符比例
    special_char_count = sum(len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))
                             for text in rec_texts)
    total_chars = sum(len(text) for text in rec_texts)
    special_ratio = special_char_count / total_chars if total_chars > 0 else 0
    if special_ratio > 0.3:
        issues.append(f"特殊字符比例过高: {special_ratio:.2%}")
    scores.append(max(0, 1 - special_ratio))

    # 4. 检查可读字符比例
    readable_chars = sum(len(re.findall(r'[\w\u4e00-\u9fff]', text))
                         for text in rec_texts)
    readable_ratio = readable_chars / total_chars if total_chars > 0 else 0
    scores.append(readable_ratio)

    quality_score = sum(scores) / len(scores) if scores else 0

    return {
        "quality_score": quality_score,
        "issues": issues,
        "metrics": {
            "empty_ratio": empty_ratio,
            "avg_length": avg_length if 'avg_length' in locals() else 0,
            "special_ratio": special_ratio,
            "readable_ratio": readable_ratio
        }
    }


# 手动触发模型加载（可选）
@app.route('/init', methods=['POST'])
def init_model():
    """手动初始化模型"""
    try:
        load_model()
        return jsonify({"success": True, "message": "模型加载成功"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    # 启动 Flask 服务
    app.run(host='0.0.0.0', port=5000, debug=False)
