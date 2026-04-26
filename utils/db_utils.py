# utils/db_utils.py
"""
数据库操作模块
功能：初始化数据库、保存预测记录、查询历史记录、删除记录、导出CSV等
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
import hashlib

# 数据库文件路径（存放在 data 子目录下）
import os
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "predictions.db")


# ========== 数据库初始化 ==========
def init_db():
    """初始化数据库，创建表（如果不存在）和索引"""
    # 确保数据目录存在
    os.makedirs(DB_DIR, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                nickname TEXT,
                probability REAL,
                risk_level TEXT,
                input_features TEXT,
                shap_summary TEXT
            )
        ''')
        # 创建索引以加速查询
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_records(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_nickname ON prediction_records(nickname)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_risk_level ON prediction_records(risk_level)')
    
    # 初始化密码表
    init_password_table()


def init_password_table():
    """初始化密码表"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS password (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 检查是否有密码记录
        count = conn.execute("SELECT COUNT(*) FROM password").fetchone()[0]
        if count == 0:
            # 设置默认密码为Shi1016!
            default_password = "Shi1016!"
            password_hash = hashlib.sha256(default_password.encode()).hexdigest()
            conn.execute(
                "INSERT INTO password (password_hash) VALUES (?)",
                (password_hash,)
            )


def get_password():
    """获取密码"""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT password_hash FROM password ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            return row[0]
        return None


def update_password(new_password):
    """更新密码"""
    password_hash = hashlib.sha256(new_password.encode()).hexdigest()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO password (password_hash) VALUES (?)",
            (password_hash,)
        )
    return True


def verify_password(password):
    """验证密码"""
    stored_hash = get_password()
    if not stored_hash:
        return False
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == stored_hash


# ========== 保存记录 ==========
def save_prediction(data: Dict) -> bool:
    """
    保存一条预测记录到数据库

    Args:
        data: 包含以下键的字典
            - nickname: 用户备注（可选）
            - probability: 风险概率（float）
            - risk_category: 风险类别（'有风险'/'无风险'）
            - input_features: 输入特征字典（将被JSON序列化）
            - shap_summary: SHAP摘要（可选，将被JSON序列化）
            - timestamp: datetime对象（可选，默认当前时间）

    Returns:
        是否保存成功
    """
    try:
        nickname = data.get('nickname', '')
        probability = float(data['probability'])
        risk_level = data['risk_category']
        input_features_json = json.dumps(data.get('input_features', {}), ensure_ascii=False)
        shap_summary_json = json.dumps(data.get('shap_summary', {}), ensure_ascii=False) if data.get(
            'shap_summary') else None
        timestamp = data.get('timestamp', datetime.now())

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO prediction_records (nickname, probability, risk_level, input_features, shap_summary, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (nickname, probability, risk_level, input_features_json, shap_summary_json, timestamp)
            )
        return True
    except Exception as e:
        print(f"保存记录失败: {e}")
        return False


# ========== 查询历史记录 ==========
def get_history(
        time_filter: str = "全部",
        risk_filter: str = "全部",
        nickname_search: str = "",
        limit: int = 50,
        offset: int = 0
) -> Tuple[pd.DataFrame, int]:
    """
    查询历史记录，支持筛选和分页

    Args:
        time_filter: 时间范围 ("全部", "近7天", "近30天", "近3个月")
        risk_filter: 风险等级 ("全部", "有风险", "无风险")
        nickname_search: 昵称关键词（模糊匹配）
        limit: 每页记录数
        offset: 偏移量

    Returns:
        (records_df, total_count) 记录DataFrame和总记录数（用于分页）
    """
    query = "SELECT * FROM prediction_records WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM prediction_records WHERE 1=1"
    params = []

    # 时间筛选
    if time_filter != "全部":
        now = datetime.now()
        if time_filter == "近7天":
            start_date = now - timedelta(days=7)
        elif time_filter == "近30天":
            start_date = now - timedelta(days=30)
        elif time_filter == "近3个月":
            start_date = now - timedelta(days=90)
        else:
            start_date = None

        if start_date:
            query += " AND timestamp >= ?"
            count_query += " AND timestamp >= ?"
            params.append(start_date)

    # 风险筛选
    if risk_filter != "全部":
        query += " AND risk_level = ?"
        count_query += " AND risk_level = ?"
        params.append(risk_filter)

    # 昵称搜索
    if nickname_search:
        query += " AND nickname LIKE ?"
        count_query += " AND nickname LIKE ?"
        params.append(f"%{nickname_search}%")

    # 获取总记录数
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute(count_query, params).fetchone()[0]

    # 分页查询
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    # 转换时间列
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df, total


# ========== 删除记录 ==========
def delete_record(record_id: int) -> bool:
    """
    根据ID删除单条记录

    Args:
        record_id: 记录ID

    Returns:
        是否删除成功
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_records WHERE id = ?", (record_id,))
        return True
    except Exception as e:
        print(f"删除记录失败: {e}")
        return False


def clear_all_records() -> bool:
    """清空所有记录"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_records")
        return True
    except Exception as e:
        print(f"清空记录失败: {e}")
        return False


# ========== 导出数据 ==========
def export_to_csv(records_df: pd.DataFrame, include_shap: bool = False) -> bytes:
    """
    将记录DataFrame导出为CSV字节流（用于下载）

    Args:
        records_df: 查询得到的DataFrame
        include_shap: 是否包含SHAP摘要列（默认False）

    Returns:
        CSV文件的字节数据，若失败返回None
    """
    if records_df.empty:
        return None

    # 选择要导出的列
    export_columns = ['id', 'timestamp', 'nickname', 'probability', 'risk_level', 'input_features']
    if include_shap:
        export_columns.append('shap_summary')

    available_cols = [col for col in export_columns if col in records_df.columns]
    df_export = records_df[available_cols].copy()

    # 格式化时间
    if 'timestamp' in df_export.columns:
        df_export['timestamp'] = pd.to_datetime(df_export['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # 可选：将JSON字符串解析为可读文本（简化处理，保持原样）
    # 如果需要更友好的展示，可以在这里做额外处理

    csv_data = df_export.to_csv(index=False, encoding='utf-8-sig')
    return csv_data.encode('utf-8-sig')


# ========== 辅助函数 ==========
def get_record_by_id(record_id: int) -> Optional[Dict]:
    """
    根据ID获取单条记录详情

    Args:
        record_id: 记录ID

    Returns:
        记录字典，若不存在返回None
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM prediction_records WHERE id = ?",
            (record_id,)
        ).fetchone()
        if row:
            return dict(row)
        return None


def get_statistics() -> Dict:
    """
    获取统计信息（用于历史记录页面）

    Returns:
        统计字典: {
            'total': 总记录数,
            'risk_count': 有风险记录数,
            'avg_probability': 平均风险概率,
            'latest_date': 最新记录日期,
            'risk_rate': 有风险比例
        }
    """
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM prediction_records").fetchone()[0]
        risk_count = conn.execute("SELECT COUNT(*) FROM prediction_records WHERE risk_level='有风险'").fetchone()[0]
        avg_prob = conn.execute("SELECT AVG(probability) FROM prediction_records").fetchone()[0]
        latest = conn.execute("SELECT MAX(timestamp) FROM prediction_records").fetchone()[0]

    risk_rate = risk_count / total if total > 0 else 0
    return {
        'total': total,
        'risk_count': risk_count,
        'avg_probability': avg_prob if avg_prob else 0.0,
        'latest_date': latest,
        'risk_rate': risk_rate
    }


# ========== 测试函数 ==========
def test_db():
    """测试数据库操作"""
    init_db()

    # 测试插入
    test_data = {
        'nickname': '测试用户',
        'probability': 0.65,
        'risk_category': '有风险',
        'input_features': {'age': 65, 'gender': '女'},
        'shap_summary': None,
        'timestamp': datetime.now()
    }
    save_prediction(test_data)
    print("插入测试记录成功")

    # 测试查询
    df, total = get_history(time_filter="全部", limit=10, offset=0)
    print(f"查询到 {total} 条记录")
    print(df.head())

    # 测试统计
    stats = get_statistics()
    print("统计信息:", stats)


if __name__ == "__main__":
    test_db()