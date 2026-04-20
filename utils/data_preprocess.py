"""
数据预处理模块
功能：验证用户输入、编码分类特征、生成模型输入特征向量
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')

# ========== 常量定义 ==========
# 特征顺序（与训练时完全一致）
FEATURE_ORDER = [
    'self_rated_health', 'pain_site_count', 'sleep_hours_night',
    'IADL_total', 'ADL_total', 'edu_years', 'childhood_health',
    'residence_type', 'gender', 'chronic_count',
    'stomach_arthritis_pair', 'arthritis_asthma_pair', 'age'
]

# 特征中文名称映射
FEATURE_NAMES_CN = {
    'self_rated_health': '自评健康',
    'pain_site_count': '疼痛部位数量',
    'sleep_hours_night': '夜间睡眠时长',
    'IADL_total': 'IADL总分',
    'ADL_total': 'ADL总分',
    'edu_years': '受教育年限',
    'childhood_health': '童年健康',
    'residence_type': '居住地类型',
    'gender': '性别',
    'chronic_count': '慢性病数量',
    'stomach_arthritis_pair': '胃病-关节炎共病',
    'arthritis_asthma_pair': '关节炎-哮喘共病',
    'age': '年龄'
}

# 选项映射
EDUCATION_MAP = {
    "未上过学": 0,
    "小学（未毕业）": 3,
    "小学": 6,
    "初中": 9,
    "高中/中专/技校": 12,
    "大专": 15,
    "本科": 16,
    "硕士": 19,
    "博士": 22
}

HEALTH_MAP = {
    "很好": 1,
    "好": 2,
    "一般": 3,
    "不好": 4,
    "很不好": 5
}

CHILDHOOD_HEALTH_MAP = {
    "极好": 1,
    "很好": 2,
    "好": 3,
    "一般": 4,
    "不好": 5
}

GENDER_MAP = {"男": 0, "女": 1}
RESIDENCE_MAP = {"城镇": 1, "农村": 0}

# 下拉框选项
HEALTH_OPTIONS = ["很好", "好", "一般", "不好", "很不好"]
HEALTH_OPTIONS_CHILDHOOD = ["极好", "很好", "好", "一般", "不好"]
GENDER_OPTIONS = ["男", "女"]
RESIDENCE_OPTIONS = ["城镇", "农村"]
EDUCATION_OPTIONS = list(EDUCATION_MAP.keys())

# 疼痛部位选项
PAIN_SITES = [
    "头", "肩", "臂", "腕", "手指", "胸", "胃",
    "背", "腰", "臀", "腿", "膝", "踝", "脚趾", "颈", "其他"
]

# 慢性病选项
CHRONIC_DISEASES = [
    "高血压", "血脂异常", "糖尿病", "癌症", "慢性肺病",
    "肝病", "心脏病", "中风", "肾病", "胃病",
    "情感及精神问题", "记忆相关疾病", "关节炎或风湿病", "哮喘"
]

# 值范围限制
VALUE_RANGES = {
    'age': (45, 120, "年龄"),
    'ADL_total': (0, 90, "ADL总分"),
    'IADL_total': (0, 6, "IADL总分"),
    'sleep_hours_night': (0, 12, "睡眠时长"),
    'pain_site_count': (0, 16, "疼痛部位数量"),
    'chronic_count': (0, 14, "慢性病数量"),
    'self_rated_health': (1, 5, "自评健康评分"),
    'childhood_health': (1, 5, "童年健康评分"),
    'edu_years': (0, 22, "受教育年限"),
    'gender': (0, 1, "性别编码"),
    'residence_type': (0, 1, "居住地编码")
}


# ========== 验证函数 ==========
def validate_input(form_data: Dict) -> List[str]:
    """
    验证用户输入的有效性

    Args:
        form_data: 包含所有表单字段的字典

    Returns:
        错误信息列表（空列表表示无错误）
    """
    errors = []

    # 1. 验证必填项
    required_fields = ['age', 'gender', 'education_level', 'residence_type',
                       'self_rated_health', 'childhood_health',
                       'ADL_total', 'IADL_total', 'sleep_hours_night']

    for field in required_fields:
        if field not in form_data or form_data[field] is None or form_data[field] == '':
            errors.append(f"请填写{get_field_name_cn(field)}")

    if errors:  # 如果有必填项缺失，先返回
        return errors

    # 2. 验证下拉框选项
    dropdown_validations = [
        ('gender', GENDER_OPTIONS, "性别"),
        ('education_level', EDUCATION_OPTIONS, "教育程度"),
        ('residence_type', RESIDENCE_OPTIONS, "居住地类型"),
        ('self_rated_health', HEALTH_OPTIONS, "自评健康"),
        ('childhood_health', HEALTH_OPTIONS_CHILDHOOD, "童年健康"),
    ]

    for field, valid_options, field_name in dropdown_validations:
        value = form_data.get(field)
        if value is not None and value not in valid_options:
            errors.append(f"{field_name}选项无效")

    # 3. 验证数值范围
    numeric_validations = [
        ('age', 'age', 45, 120, "岁"),
        ('ADL_total', 'ADL_total', 0, 90, ""),
        ('IADL_total', 'IADL_total', 0, 6, ""),
        ('sleep_hours_night', 'sleep_hours_night', 0, 12, "小时"),
    ]

    for field, display_name, min_val, max_val, unit in numeric_validations:
        value = form_data.get(field)
        if value is not None:
            try:
                num_value = float(value)
                if num_value < min_val or num_value > max_val:
                    errors.append(f"{get_field_name_cn(display_name)}必须在{min_val}-{max_val}{unit}之间")
            except (ValueError, TypeError):
                errors.append(f"{get_field_name_cn(display_name)}必须是数字")

    # 4. 验证列表字段
    list_fields = [
        ('pain_sites', "疼痛部位"),
        ('chronic_diseases', "慢性病")
    ]

    for field, field_name in list_fields:
        value = form_data.get(field, [])
        if not isinstance(value, list):
            errors.append(f"{field_name}必须是列表")
        elif field == 'chronic_diseases':
            # 验证慢性病选项是否有效
            for disease in value:
                if disease not in CHRONIC_DISEASES:
                    errors.append(f"慢性病选项'{disease}'无效")

    # 5. 验证布尔字段
    bool_fields = ['stomach_arthritis_pair', 'arthritis_asthma_pair']
    for field in bool_fields:
        value = form_data.get(field)
        if value is not None and not isinstance(value, bool):
            errors.append(f"{get_field_name_cn(field)}必须是True或False")

    return errors


def get_field_name_cn(field: str) -> str:
    """获取字段的中文名"""
    field_names = {
        'age': '年龄',
        'gender': '性别',
        'education_level': '教育程度',
        'residence_type': '居住地类型',
        'self_rated_health': '自评健康',
        'childhood_health': '童年健康',
        'ADL_total': 'ADL总分',
        'IADL_total': 'IADL总分',
        'sleep_hours_night': '睡眠时长',
        'pain_sites': '疼痛部位',
        'chronic_diseases': '慢性病',
        'stomach_arthritis_pair': '胃病-关节炎共病',
        'arthritis_asthma_pair': '关节炎-哮喘共病'
    }
    return field_names.get(field, field)


# ========== 特征编码函数 ==========
def encode_categorical_features(form_data: Dict) -> Dict:
    """
    将分类特征编码为数值

    Args:
        form_data: 包含原始表单数据的字典

    Returns:
        编码后的数值特征字典
    """
    encoded = {}

    # 基本特征直接映射或计算
    encoded['age'] = float(form_data.get('age', 65))
    encoded['gender'] = GENDER_MAP.get(form_data.get('gender', '男'), 0)
    encoded['residence_type'] = RESIDENCE_MAP.get(form_data.get('residence_type', '农村'), 0)

    # 健康状态
    encoded['self_rated_health'] = HEALTH_MAP.get(form_data.get('self_rated_health', '好'), 3)
    encoded['childhood_health'] = CHILDHOOD_HEALTH_MAP.get(form_data.get('childhood_health', '好'), 3)

    # 功能状态
    encoded['ADL_total'] = float(form_data.get('ADL_total', 90))
    encoded['IADL_total'] = float(form_data.get('IADL_total', 6))

    # 生活方式
    encoded['sleep_hours_night'] = float(form_data.get('sleep_hours_night', 7.0))

    # 疼痛部位数量
    pain_sites = form_data.get('pain_sites', [])
    encoded['pain_site_count'] = len(pain_sites) if isinstance(pain_sites, list) else 0

    # 受教育年限
    edu_level = form_data.get('education_level', '初中')
    encoded['edu_years'] = EDUCATION_MAP.get(edu_level, 9)

    # 慢性病数量
    chronic_diseases = form_data.get('chronic_diseases', [])
    encoded['chronic_count'] = len(chronic_diseases) if isinstance(chronic_diseases, list) else 0

    # 共病簇
    encoded['stomach_arthritis_pair'] = 1 if form_data.get('stomach_arthritis_pair', False) else 0
    encoded['arthritis_asthma_pair'] = 1 if form_data.get('arthritis_asthma_pair', False) else 0

    return encoded


def preprocess_input(form_data: Dict) -> pd.DataFrame:
    """
    主预处理函数：将前端表单数据转换为模型可用的DataFrame

    Args:
        form_data: 包含所有表单字段的字典

    Returns:
        形状为(1, 13)的DataFrame，列顺序与训练时一致
    """
    # 1. 编码分类特征
    encoded = encode_categorical_features(form_data)

    # 2. 确保所有特征都存在（若缺失则使用默认值）
    default_values = {
        'self_rated_health': 3,
        'childhood_health': 3,
        'pain_site_count': 0,
        'sleep_hours_night': 7.0,
        'IADL_total': 6,
        'ADL_total': 90,
        'edu_years': 9,
        'residence_type': 0,
        'gender': 0,
        'chronic_count': 0,
        'stomach_arthritis_pair': 0,
        'arthritis_asthma_pair': 0,
        'age': 65
    }

    for feature in FEATURE_ORDER:
        if feature not in encoded:
            encoded[feature] = default_values.get(feature, 0)

    # 3. 按顺序构建DataFrame
    feature_values = [encoded[feature] for feature in FEATURE_ORDER]
    df = pd.DataFrame([feature_values], columns=FEATURE_ORDER)

    return df


def encode_categorical_features_with_raw(form_data: Dict) -> Tuple[Dict, Dict]:
    """
    将分类特征编码为数值，并返回原始特征

    Args:
        form_data: 包含原始表单数据的字典

    Returns:
        (编码后的特征字典, 原始特征字典)
    """
    encoded = encode_categorical_features(form_data)

    # 创建原始特征字典用于展示
    raw_features = {
        'age': form_data.get('age', 65),
        'gender': form_data.get('gender', '男'),
        'education_level': form_data.get('education_level', '初中'),
        'education_years': EDUCATION_MAP.get(form_data.get('education_level', '初中'), 9),
        'residence_type': form_data.get('residence_type', '农村'),
        'self_rated_health': form_data.get('self_rated_health', '好'),
        'childhood_health': form_data.get('childhood_health', '好'),
        'pain_site_count': len(form_data.get('pain_sites', [])),
        'pain_sites': form_data.get('pain_sites', []),
        'ADL_total': form_data.get('ADL_total', 90),
        'IADL_total': form_data.get('IADL_total', 6),
        'chronic_count': len(form_data.get('chronic_diseases', [])),
        'chronic_diseases': form_data.get('chronic_diseases', []),
        'stomach_arthritis_pair': form_data.get('stomach_arthritis_pair', False),
        'arthritis_asthma_pair': form_data.get('arthritis_asthma_pair', False),
        'sleep_hours_night': form_data.get('sleep_hours_night', 7.0)
    }

    return encoded, raw_features


# ========== 辅助工具函数 ==========
def update_comorbidities(selected_diseases: List[str]) -> Tuple[bool, bool]:
    """
    根据选择的慢性病列表自动更新共病簇状态

    Args:
        selected_diseases: 选择的慢性病名称列表

    Returns:
        (stomach_arthritis_pair, arthritis_asthma_pair)
    """
    stomach_arthritis = ("胃病" in selected_diseases and "关节炎或风湿病" in selected_diseases)
    arthritis_asthma = ("关节炎或风湿病" in selected_diseases and "哮喘" in selected_diseases)
    return stomach_arthritis, arthritis_asthma


def get_feature_names() -> List[str]:
    """返回特征顺序列表（用于模型输入）"""
    return FEATURE_ORDER.copy()


def get_feature_names_cn() -> Dict[str, str]:
    """返回特征中文名称映射（用于显示）"""
    return FEATURE_NAMES_CN.copy()


def calculate_statistics(form_data: Dict) -> Dict:
    """
    计算输入数据的统计信息

    Args:
        form_data: 原始表单数据

    Returns:
        统计信息字典
    """
    stats = {}

    # 年龄分组
    age = float(form_data.get('age', 65))
    stats['age_group'] = '中年(45-59)' if age < 60 else '老年(60+)'
    stats['gender'] = form_data.get('gender', '女')

    # ADL功能状态
    adl_score = float(form_data.get('ADL_total', 90))
    if adl_score >= 60:
        stats['adl_status'] = '良好'
    elif adl_score >= 30:
        stats['adl_status'] = '一般'
    else:
        stats['adl_status'] = '较差'

    # IADL功能状态
    iadl_score = float(form_data.get('IADL_total', 6))
    stats['iadl_status'] = '良好' if iadl_score >= 4 else '困难'

    # 健康状态评分
    health_map = {'很好': 1, '好': 2, '一般': 3, '不好': 4, '很不好': 5}
    self_health = form_data.get('self_rated_health', '好')
    stats['self_health_score'] = health_map.get(self_health, 3)

    # 睡眠质量
    sleep_hours = float(form_data.get('sleep_hours_night', 7.0))
    if 7 <= sleep_hours <= 9:
        stats['sleep_quality'] = '良好'
    elif 5 <= sleep_hours < 7:
        stats['sleep_quality'] = '一般'
    else:
        stats['sleep_quality'] = '较差'

    # 慢性病负担
    chronic_count = len(form_data.get('chronic_diseases', []))
    if chronic_count == 0:
        stats['chronic_burden'] = '无'
    elif chronic_count <= 2:
        stats['chronic_burden'] = '轻度'
    elif chronic_count <= 4:
        stats['chronic_burden'] = '中度'
    else:
        stats['chronic_burden'] = '重度'

    # 疼痛负担
    pain_count = len(form_data.get('pain_sites', []))
    if pain_count == 0:
        stats['pain_burden'] = '无'
    elif pain_count <= 2:
        stats['pain_burden'] = '轻度'
    elif pain_count <= 4:
        stats['pain_burden'] = '中度'
    else:
        stats['pain_burden'] = '重度'

    return stats


def format_feature_for_display(feature_name: str, feature_value, raw_value=None) -> str:
    """
    格式化特征值用于展示

    Args:
        feature_name: 特征名称
        feature_value: 编码后的特征值
        raw_value: 原始值（可选）

    Returns:
        格式化的特征描述
    """
    cn_name = FEATURE_NAMES_CN.get(feature_name, feature_name)

    if raw_value is not None:
        display_value = raw_value
    else:
        display_value = feature_value

    if feature_name == 'gender':
        return f"{cn_name}: {'女性' if feature_value == 1 else '男性'}"

    elif feature_name == 'residence_type':
        return f"{cn_name}: {'城市' if feature_value == 1 else '农村'}"

    elif feature_name == 'self_rated_health':
        options = ["很好", "好", "一般", "不好", "很不好"]
        idx = int(feature_value) - 1
        if 0 <= idx < len(options):
            return f"{cn_name}: {options[idx]}"

    elif feature_name == 'childhood_health':
        options = ["极好", "很好", "好", "一般", "不好"]
        idx = int(feature_value) - 1
        if 0 <= idx < len(options):
            return f"{cn_name}: {options[idx]}"

    elif feature_name in ['stomach_arthritis_pair', 'arthritis_asthma_pair']:
        condition = "胃病-关节炎共病" if feature_name == 'stomach_arthritis_pair' else "关节炎-哮喘共病"
        return f"{cn_name}: {'是' if feature_value == 1 else '否'}"

    elif feature_name in ['pain_site_count', 'chronic_count']:
        if isinstance(display_value, (list, tuple)):
            count = len(display_value)
            items = ', '.join(display_value) if display_value else '无'
            return f"{cn_name}: {count}个 ({items})"
        else:
            return f"{cn_name}: {int(feature_value)}个"

    elif feature_name == 'edu_years':
        return f"{cn_name}: {int(feature_value)}年"

    elif feature_name == 'ADL_total':
        score = int(feature_value)
        status = "良好" if score >= 60 else "一般" if score >= 30 else "较差"
        return f"{cn_name}: {score}/90 ({status})"

    elif feature_name == 'IADL_total':
        score = int(feature_value)
        status = "良好" if score >= 4 else "困难"
        return f"{cn_name}: {score}/6 ({status})"

    elif feature_name == 'sleep_hours_night':
        hours = float(feature_value)
        if 7 <= hours <= 9:
            quality = "良好"
        elif 5 <= hours < 7:
            quality = "一般"
        else:
            quality = "较差"
        return f"{cn_name}: {hours:.1f}小时 ({quality})"

    elif feature_name == 'age':
        age_group = "中年(45-59)" if feature_value < 60 else "老年(60+)"
        return f"{cn_name}: {int(feature_value)}岁 ({age_group})"

    else:
        return f"{cn_name}: {display_value}"


def get_feature_importance_order(shap_values: np.ndarray, feature_names: List[str]) -> List[Tuple[str, float]]:
    """
    根据SHAP值获取特征重要性排序

    Args:
        shap_values: SHAP值数组
        feature_names: 特征名称列表

    Returns:
        按重要性排序的特征列表
    """
    if shap_values is None or len(shap_values) == 0:
        return []

    if len(shap_values.shape) == 2:
        shap_importance = np.abs(shap_values).mean(axis=0)
    else:
        shap_importance = np.abs(shap_values)

    feature_importance = []
    for i, feature in enumerate(feature_names):
        if i < len(shap_importance):
            feature_importance.append((feature, float(shap_importance[i])))

    feature_importance.sort(key=lambda x: x[1], reverse=True)
    return feature_importance


# ========== 测试函数 ==========
def test_preprocess():
    """测试预处理函数"""
    test_data = {
        'age': 65,
        'gender': '女',
        'education_level': '初中',
        'residence_type': '城镇',
        'self_rated_health': '好',
        'childhood_health': '好',
        'pain_sites': ['头', '背'],
        'sleep_hours_night': 7.0,
        'ADL_total': 85,
        'IADL_total': 5,
        'chronic_diseases': ['高血压', '糖尿病'],
        'stomach_arthritis_pair': False,
        'arthritis_asthma_pair': False
    }

    print("测试数据验证:")
    errors = validate_input(test_data)
    print("验证结果:", "通过" if not errors else errors)

    print("\n编码后的特征:")
    encoded = encode_categorical_features(test_data)
    for key, value in encoded.items():
        cn_name = FEATURE_NAMES_CN.get(key, key)
        print(f"  {cn_name}: {value}")

    print("\n原始特征:")
    _, raw = encode_categorical_features_with_raw(test_data)
    for key, value in raw.items():
        print(f"  {key}: {value}")

    print("\n模型输入:")
    df = preprocess_input(test_data)
    print(df)

    print("\n统计信息:")
    stats = calculate_statistics(test_data)
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_preprocess()