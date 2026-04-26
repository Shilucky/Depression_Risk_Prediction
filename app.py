"""
抑郁风险预警系统 - Streamlit Web应用
主程序文件
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import json
import os
import time
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
from io import BytesIO
import base64
import uuid
import toml


def set_matplotlib_chinese_font():
    """自动设置支持中文的字体"""
    # 尝试查找可用的中文字体
    font_list = [
        'WenQuanYi Micro Hei',    # Linux 文泉驿
        'Noto Sans CJK SC',       # 思源黑体
        'SimHei',                 # Windows
        'STHeiti',                # macOS
        'Microsoft YaHei'         # Windows 雅黑
    ]
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    for font in font_list:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            break
    else:
        # 如果都没找到，尝试使用系统默认中文字体（可能失败）
        plt.rcParams['font.sans-serif'] = ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


# 加载配置文件
def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.toml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        return config
    except Exception as e:
        st.error(f"加载配置文件失败: {str(e)}")
        # 返回默认配置
        return {
            'risk_threshold': 0.492,
            'model': {'model_path': 'models/CatBoost_F2_v1_with_threshold.pkl'},
            'shap': {
                'local_shap_dir': 'shap_results',
                'original_shap_dir': 'D:\\AAA毕业\\AAA毕业设计\\3Ending\\主实验-二分类\\机器学习-二分类\\output\\evaluation\\shap',
                'local_test_data_path': 'shap_results/X_test.csv',
                'original_test_data_path': 'D:\\AAA毕业\\AAA毕业设计\\3Ending\\主实验-二分类\\机器学习-二分类\\output\\data\\X_test.csv'
            },
            'default_values': {
                'age': 65,
                'gender': '女',
                'education_level': '初中',
                'residence_type': '城镇',
                'self_rated_health': '好',
                'childhood_health': '好',
                'pain_sites': [],
                'sleep_hours_night': 7.0,
                'ADL_total': 90,
                'IADL_total': 6,
                'chronic_diseases': [],
                'stomach_arthritis_pair': False,
                'arthritis_asthma_pair': False
            },
            'features': {
                'feature_order': [
                    'self_rated_health', 'pain_site_count', 'sleep_hours_night',
                    'IADL_total', 'ADL_total', 'edu_years', 'childhood_health',
                    'residence_type', 'gender', 'chronic_count',
                    'stomach_arthritis_pair', 'arthritis_asthma_pair', 'age'
                ]
            },
            'feature_names_cn': {
                'gender': '性别',
                'age': '年龄',
                'edu_years': '受教育年限',
                'residence_type': '居住地类型',
                'self_rated_health': '自评健康',
                'childhood_health': '童年健康',
                'pain_site_count': '疼痛部位数量',
                'ADL_total': 'ADL总分',
                'IADL_total': 'IADL总分',
                'chronic_count': '慢性病数量',
                'stomach_arthritis_pair': '胃病-关节炎共病',
                'arthritis_asthma_pair': '关节炎-哮喘共病',
                'sleep_hours_night': '夜间睡眠时长'
            },
            'education_map': {
                '未上过学': 0,
                '小学（未毕业）': 3,
                '小学': 6,
                '初中': 9,
                '高中/中专/技校': 12,
                '大专': 15,
                '本科': 16,
                '硕士': 19,
                '博士': 22
            },
            'options': {
                'health_options': ['很好', '好', '一般', '不好', '很不好'],
                'health_options_childhood': ['极好', '很好', '好', '一般', '不好'],
                'pain_sites': [
                    '头', '肩', '臂', '腕', '手指', '胸', '胃',
                    '背', '腰', '臀', '腿', '膝', '踝', '脚趾', '颈', '其他'
                ],
                'chronic_diseases': [
                    '高血压', '血脂异常', '糖尿病', '癌症', '慢性肺病',
                    '肝病', '心脏病', '中风', '肾病', '胃病',
                    '情感及精神问题', '记忆相关疾病', '关节炎或风湿病', '哮喘'
                ]
            },
            'comorbidity_map': {
                'stomach_arthritis': ['胃病', '关节炎或风湿病'],
                'arthritis_asthma': ['关节炎或风湿病', '哮喘']
            }
        }

# 加载配置
CONFIG = load_config()

# 从Streamlit Secrets读取部署配置
DEPLOY_MODE = st.secrets.get("DEPLOY_MODE", "local")

# 密码验证函数
def verify_admin_password(password):
    """验证管理员密码"""
    from utils.db_utils import verify_password
    return verify_password(password)

# 更新密码函数
def update_admin_password(current_password, new_password):
    """更新管理员密码"""
    from utils.db_utils import verify_password, update_password
    if not verify_password(current_password):
        return False, "当前密码错误"
    if len(new_password) < 8:
        return False, "新密码长度至少为8位"
    if update_password(new_password):
        return True, "密码修改成功"
    return False, "密码修改失败"

# 在程序入口调用
set_matplotlib_chinese_font()

# 从配置文件获取主题设置
theme_config = CONFIG.get('theme', {})
theme_base = theme_config.get('base', 'light')
theme_primary_color = theme_config.get('primaryColor', '#1a73e8')

# 设置页面配置
st.set_page_config(
    page_title=CONFIG.get('page', {}).get('title', "抑郁风险预警系统"),
    page_icon="🧠",
    layout=CONFIG.get('page', {}).get('layout', "wide"),
    initial_sidebar_state=CONFIG.get('page', {}).get('initial_sidebar_state', "expanded"),
    menu_items={
        'Get Help': 'https://github.com/yourusername/depression-risk-app',
        'Report a bug': None,
        'About': f"""
        ## {CONFIG.get('page', {}).get('title', "抑郁风险预警系统")}
        基于CatBoost模型的社区中老年人抑郁风险筛查工具
        版本: {CONFIG.get('version', {}).get('number', "1.0.0")}
        开发团队: 公共卫生研究团队
        """
    }
)

# 设置主题
st.markdown(
    f"""
    <style>
    /* 主题设置 */
    :root {{
        --primary-color: {theme_primary_color};
    }}
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {{
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }}
    
    /* 标题样式 */
    [data-testid="stSidebar"] h1 {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {theme_primary_color};
        margin-bottom: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e3f2fd;
        text-align: center;
    }}
    
    /* 导航菜单项样式 */
    [data-testid="stSidebar"] button[kind="primary"] {{
        background: linear-gradient(135deg, {theme_primary_color} 0%, #1557b0 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 3px 6px rgba(26, 115, 232, 0.25) !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }}
    
    /* 主内容区Primary按钮样式 */
    div[data-testid="stMain"] button[kind="primary"],
    section.main button[kind="primary"],
    .stButton > button[kind="primary"] {{
        background-color: {theme_primary_color} !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(26, 115, 232, 0.2) !important;
    }}
    
    /* Number Input +/- 按钮颜色 */
    div[data-testid="stNumberInput"] button {{
        color: {theme_primary_color} !important;
        border-color: {theme_primary_color} !important;
    }}
    
    /* Radio按钮的外圈和内部圆点 */
    .stRadio > div[role="radiogroup"] > label > div:first-child {{
        border-color: {theme_primary_color} !important;
    }}
    
    .stRadio > div[role="radiogroup"] > label:has(input:checked) > div:first-child {{
        background-color: {theme_primary_color} !important;
        border-color: {theme_primary_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 导入工具模块
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.data_preprocess import (
        preprocess_input, validate_input, encode_categorical_features, update_comorbidities,
        FEATURE_NAMES_CN, FEATURE_ORDER, EDUCATION_MAP,
        HEALTH_OPTIONS, HEALTH_OPTIONS_CHILDHOOD, PAIN_SITES, CHRONIC_DISEASES
    )
    from utils.db_utils import (
        init_db, save_prediction, get_history, get_statistics,
        delete_record, clear_all_records, export_to_csv
    )
    from utils.shap_utils import (
        generate_shap_values, create_shap_waterfall_plot,
        generate_text_explanation, get_feature_names_cn,
        create_interactive_shap_waterfall, generate_feature_importance_bar,
        generate_dependence_plot
    )
except ImportError as e:
    st.warning(f"工具模块未找到: {str(e)}，请确保utils目录存在且包含相应文件")


    # 定义临时函数避免错误
    def preprocess_input(*args, **kwargs):
        return {}


    def validate_input(*args, **kwargs):
        return []


    def encode_categorical_features(*args, **kwargs):
        return {}


    def init_db():
        pass


    def save_prediction(*args, **kwargs):
        return None


    def get_history(*args, **kwargs):
        return pd.DataFrame()


    def delete_record(*args, **kwargs):
        pass


    def clear_all_records():
        pass


    def export_to_csv(*args, **kwargs):
        return None


    def get_statistics():
        return {'total': 0, 'risk_count': 0, 'avg_probability': 0.0, 'latest_date': None, 'risk_rate': 0.0}


    def generate_shap_values(*args, **kwargs):
        return (None, None)


    def create_shap_waterfall_plot(*args, **kwargs):
        return None


    def generate_text_explanation(*args, **kwargs):
        return ""


    def get_feature_names_cn():
        return {}


    def create_interactive_shap_waterfall(*args, **kwargs):
        import plotly.graph_objects as go
        return go.Figure()

    def update_comorbidities(selected_diseases):
        """更新共病簇状态"""
        stomach_arthritis = False
        arthritis_asthma = False

        if "胃病" in selected_diseases and "关节炎或风湿病" in selected_diseases:
            stomach_arthritis = True
        if "关节炎或风湿病" in selected_diseases and "哮喘" in selected_diseases:
            arthritis_asthma = True

        return stomach_arthritis, arthritis_asthma

    # 定义临时常量避免错误
    FEATURE_NAMES_CN = CONFIG.get('feature_names_cn', {
        'gender': '性别',
        'age': '年龄',
        'edu_years': '受教育年限',
        'residence_type': '居住地类型',
        'self_rated_health': '自评健康',
        'childhood_health': '童年健康',
        'pain_site_count': '疼痛部位数量',
        'ADL_total': 'ADL总分',
        'IADL_total': 'IADL总分',
        'chronic_count': '慢性病数量',
        'stomach_arthritis_pair': '胃病-关节炎共病',
        'arthritis_asthma_pair': '关节炎-哮喘共病',
        'sleep_hours_night': '夜间睡眠时长'
    })

    FEATURE_ORDER = CONFIG.get('features', {}).get('feature_order', [
        'self_rated_health', 'pain_site_count', 'sleep_hours_night',
        'IADL_total', 'ADL_total', 'edu_years', 'childhood_health',
        'residence_type', 'gender', 'chronic_count',
        'stomach_arthritis_pair', 'arthritis_asthma_pair', 'age'
    ])

    EDUCATION_MAP = CONFIG.get('education_map', {
        "未上过学": 0,
        "小学（未毕业）": 3,
        "小学": 6,
        "初中": 9,
        "高中/中专/技校": 12,
        "大专": 15,
        "本科": 16,
        "硕士": 19,
        "博士": 22
    })

    HEALTH_OPTIONS = CONFIG.get('options', {}).get('health_options', ["很好", "好", "一般", "不好", "很不好"])
    HEALTH_OPTIONS_CHILDHOOD = CONFIG.get('options', {}).get('health_options_childhood', ["极好", "很好", "好", "一般", "不好"])

    PAIN_SITES = CONFIG.get('options', {}).get('pain_sites', [
        "头", "肩", "臂", "腕", "手指", "胸", "胃",
        "背", "腰", "臀", "腿", "膝", "踝", "脚趾", "颈", "其他"
    ])

    CHRONIC_DISEASES = CONFIG.get('options', {}).get('chronic_diseases', [
        "高血压", "血脂异常", "糖尿病", "癌症", "慢性肺病",
        "肝病", "心脏病", "中风", "肾病", "胃病",
        "情感及精神问题", "记忆相关疾病", "关节炎或风湿病", "哮喘"
    ])

# 共病簇映射
COMORBIDITY_MAP = CONFIG.get('comorbidity_map', {
    "stomach_arthritis": ["胃病", "关节炎或风湿病"],
    "arthritis_asthma": ["关节炎或风湿病", "哮喘"]
})

# 风险阈值
RISK_THRESHOLD = CONFIG.get('risk_threshold', 0.492)  # CatBoost模型最优阈值

# 默认值
DEFAULT_VALUES = CONFIG.get('default_values', {
    'age': 65,
    'gender': '女',
    'education_level': '初中',
    'residence_type': '城镇',
    'self_rated_health': '好',
    'childhood_health': '好',
    'pain_sites': [],
    'sleep_hours_night': 7.0,
    'ADL_total': 90,
    'IADL_total': 6,
    'chronic_diseases': [],
    'stomach_arthritis_pair': False,
    'arthritis_asthma_pair': False
})


# ========== 工具函数 ==========







def load_model():
    """加载模型和SHAP解释器"""
    try:
        import os
        # 从配置文件获取模型路径
        model_path = CONFIG.get('model', {}).get('model_path', 'models/CatBoost_F2_v1_with_threshold.pkl')
        # 使用绝对路径加载模型文件
        model_path = os.path.join(os.path.dirname(__file__), model_path)
        model_data = joblib.load(model_path)

        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            threshold = model_data.get('threshold', RISK_THRESHOLD)
        else:
            # 假设直接是模型对象
            model = model_data
            threshold = RISK_THRESHOLD

        # 从Pipeline中提取模型（如果是Pipeline）
        if hasattr(model, 'named_steps'):
            # 尝试获取最后一步的模型
            for step_name in reversed(model.named_steps):
                step_model = model.named_steps[step_name]
                # 检查是否是CatBoost模型或其他树模型
                if hasattr(step_model, 'predict_proba'):
                    model = step_model
                    break

        # 使用手动定义的特征顺序（与训练时一致）
        feature_names = FEATURE_ORDER

        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)

        return {
            'model': model,
            'explainer': explainer,
            'threshold': threshold,
            'feature_names': feature_names
        }
    except FileNotFoundError:
        st.error("模型文件未找到，请确保 models/CatBoost_F2_v1_with_threshold.pkl 存在")
        st.stop()
    except Exception as e:
        st.error(f"加载模型时发生错误: {str(e)}")
        st.stop()





def render_adl_popover():
    """渲染ADL速测弹出框"""
    # 直接使用st.popover创建速测按钮
    with st.popover("ADL速测"):
        st.markdown("### ADL（日常生活活动能力）评估")
        st.info("请根据实际情况选择，系统会自动计算总分。")

        adl_items = {
            "进食": [("独立", 10), ("部分帮助", 5), ("大量帮助", 0), ("完全依赖", 0)],
            "洗澡": [("独立", 5), ("部分帮助", 0), ("大量帮助", 0), ("完全依赖", 0)],
            "穿衣": [("独立", 10), ("部分帮助", 5), ("大量帮助", 0), ("完全依赖", 0)],
            "如厕": [("独立", 10), ("部分帮助", 5), ("大量帮助", 0), ("完全依赖", 0)],
            "控制大小便": [("独立", 10), ("部分帮助", 5), ("大量帮助", 0), ("完全依赖", 0)],
            "床椅转移": [("独立", 15), ("部分帮助", 10), ("大量帮助", 5), ("完全依赖", 0)],
            "平地行走": [("独立", 15), ("部分帮助", 10), ("大量帮助", 5), ("完全依赖", 0)],
            "上下楼梯": [("独立", 10), ("部分帮助", 5), ("大量帮助", 0), ("完全依赖", 0)],
            "修饰（精细动作）": [("独立", 5), ("部分帮助", 0), ("大量帮助", 0), ("完全依赖", 0)]
        }

        scores = {}
        for item, options in adl_items.items():
            options_text = [f"{opt} ({score}分)" for opt, score in options]
            selected = st.radio(
                f"{item}：",
                options=options_text,
                horizontal=True,
                key=f"adl_{item}"
            )
            # 提取分数
            score = options[options_text.index(selected)][1]
            scores[item] = score

        total_score = sum(scores.values())

        st.markdown('<div class="adl-apply-button">', unsafe_allow_html=True)
        if st.button("✅ 应用ADL总分", type="primary", use_container_width=True, key="adl_apply_btn"):
            st.session_state.form_data['ADL_total'] = total_score
            st.success(f"已设置ADL总分为: {total_score}")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.metric("ADL总分", total_score)


def render_iadl_popover():
    """渲染IADL速测弹出框"""
    # 直接使用st.popover创建速测按钮
    with st.popover("IADL速测"):
        st.markdown("### IADL（工具性日常生活活动能力）评估")
        st.info("请根据实际情况选择，默认全部为'无困难'。")

        iadl_items = [
            "做家务", "准备饭菜", "购物",
            "管理钱财", "服药", "打电话"
        ]

        difficulties = {}
        for item in iadl_items:
            difficulties[item] = st.checkbox(
                f"{item}有困难",
                value=False,
                key=f"iadl_{item}"
            )

        # 计算总分：无困难=1分，有困难=0分
        total_score = sum([0 if difficulties[item] else 1 for item in iadl_items])

        st.markdown('<div class="iadl-apply-button">', unsafe_allow_html=True)
        if st.button("✅ 应用IADL总分", type="primary", use_container_width=True, key="iadl_apply_btn"):
            st.session_state.form_data['IADL_total'] = total_score
            st.success(f"已设置IADL总分为: {total_score}")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.metric("IADL总分", total_score)





def get_risk_info(probability: float, threshold: float = RISK_THRESHOLD) -> Dict:
    """根据概率获取风险信息"""
    is_risk = probability >= threshold

    # 从配置中获取风险类别和颜色
    risk_config = CONFIG.get('risk', {})
    risk_categories = risk_config.get('risk_category', ["无风险", "有风险"])
    risk_colors = risk_config.get('risk_colors', ["green", "red"])

    # 只返回二分类结果
    risk_category = risk_categories[1] if is_risk else risk_categories[0]
    color = risk_colors[1] if is_risk else risk_colors[0]

    return {
        'is_risk': is_risk,
        'risk_category': risk_category,
        'color': color,
        'probability': probability
    }


def get_personalized_advice(risk_info: Dict, shap_values: np.ndarray,
                            feature_names: List[str], feature_values: Dict) -> str:
    """生成个性化建议"""
    is_risk = risk_info['is_risk']

    # 获取最重要的风险因素
    top_features = []
    if shap_values is not None and len(shap_values) > 0:
        feature_importance = np.abs(shap_values[0])
        top_indices = np.argsort(feature_importance)[-3:][::-1]
        top_features = [feature_names[i] for i in top_indices]

    advice = f"\n\n"

    if is_risk:
        advice += f"""
        ⚠️ 建议立即采取行动：
        1. 强烈建议尽快咨询专业医生或心理卫生专家
        2. 与家人或朋友沟通您的感受
        3. 社区心理卫生服务中心可提供专业支持
        4. 保持健康饮食和适度运动
        5. 避免独自承受压力
        """
    else:
        advice += f"""
        👍 当前状态良好：
        1. 继续保持健康的生活方式
        2. 定期进行自我评估
        3. 参与社区活动，保持社交
        """

    # 添加针对性建议
    if top_features:
        pass

    return advice


# ========== 页面渲染函数 ==========
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        # 添加自定义CSS
        st.markdown('''
        <!-- Font Awesome CDN -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        
        <style>
        /* 全局样式 */
        body {
            font-size: 16px !important;
        }
        
        /* 表单元素样式 */
        .stNumberInput, .stSelectbox, .stRadio, .stSlider, .stCheckbox {
            font-size: 16px !important;
        }
        
        /* 标签样式 */
        .stNumberInput label, .stSelectbox label, .stRadio label, .stSlider label, .stCheckbox label {
            font-size: 16px !important;
            font-weight: 500;
        }
        
        /* 系统信息caption样式 */
        [data-testid="stSidebar"] .stCaption {
            font-size: 5px !important;
            line-height: 1.4;
        }
        
        /* 侧边栏整体样式 */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e0e0e0;
        }
        
        /* 标题样式 */
        [data-testid="stSidebar"] h1 {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1a73e8;
            margin-bottom: 2rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e3f2fd;
            text-align: center;
        }
        
        /* 导航菜单项样式 - 终极强制覆盖版 */ 
         [data-testid="stSidebar"] button[kind="primary"] { 
             background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%) !important; 
             color: white !important; 
             border: none !important; 
             box-shadow: 0 3px 6px rgba(26, 115, 232, 0.25) !important; 
             font-weight: 600 !important; 
             letter-spacing: 0.5px; 
         } 
         
         [data-testid="stSidebar"] button[kind="primary"]:hover { 
             background: linear-gradient(135deg, #1557b0 0%, #0d47a1 100%) !important; 
             transform: translateX(4px); 
             box-shadow: 0 4px 10px rgba(26, 115, 232, 0.35) !important; 
         } 
         
         /* 强制覆盖侧边栏所有非primary按钮 */ 
         [data-testid="stSidebar"] .stButton > button, 
         [data-testid="stSidebar"] button { 
             background-color: #ffffff !important; 
             color: #5f6368 !important; 
             border: 1.5px solid #dadce0 !important; 
             font-weight: 500 !important; 
             transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important; 
         } 
         
         [data-testid="stSidebar"] .stButton > button:hover, 
         [data-testid="stSidebar"] button:hover { 
             background: linear-gradient(135deg, #e8f0fe 0%, #d2e3fc 100%) !important; 
             color: #1a73e8 !important; 
             border-color: #1a73e8 !important; 
             transform: translateX(4px); 
             box-shadow: 0 2px 6px rgba(26, 115, 232, 0.15) !important; 
         } 
         
         /* 但要保持高风险和低风险按钮的特殊样式 */ 
         [data-testid="stButton-high_risk_button"] button, 
         [data-testid="stButton-high_risk_button"] button:hover { 
             background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%) !important; 
             border: 1.5px solid #ffcdd2 !important; 
             color: #c62828 !important; 
         } 
         
         [data-testid="stButton-low_risk_button"] button, 
         [data-testid="stButton-low_risk_button"] button:hover { 
             background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%) !important; 
             border: 1.5px solid #c8e6c9 !important; 
             color: #2e7d32 !important; 
         }
        
        /* 主内容区Primary按钮样式 - 蓝色（包括开始评估按钮）*/
        div[data-testid="stMain"] button[kind="primary"],
        section.main button[kind="primary"],
        .stButton > button[kind="primary"] {
            background-color: #1a73e8 !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 2px 4px rgba(26, 115, 232, 0.2) !important;
        }
        
        div[data-testid="stMain"] button[kind="primary"]:hover,
        section.main button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:hover {
            background-color: #1557b0 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(26, 115, 232, 0.3) !important;
        }
        
        /* 重置按钮样式优化 */ 
         [data-testid="stSidebar"] button[key*="reset"], 
         div[data-testid="stSidebar"] button:has(span:contains("重置")) { 
             background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%) !important; 
             color: #616161 !important; 
             border: 1.5px solid #bdbdbd !important; 
         } 
         
         [data-testid="stSidebar"] button[key*="reset"]:hover, 
         div[data-testid="stSidebar"] button:has(span:contains("重置")):hover { 
             background: linear-gradient(135deg, #eeeeee 0%, #e0e0e0 100%) !important; 
             color: #424242 !important; 
             border-color: #9e9e9e !important; 
         }
        
        /* 大字体，适合中老年用户 */
        .stNumberInput input, .stTextInput input, .stSelectbox select, .stRadio label, .stCheckbox label, .stSlider label {
            font-size: 20px !important;
        }
        
        /* 按钮样式 */
        .stButton > button {
            font-size: 18px;
            padding: 10px 20px;
        }
        
        /* 标签文字大小 */
        .stMarkdown h3, .stMarkdown p {
            font-size: 18px !important;
        }
        
        /* Radio按钮选中状态颜色 - 蓝色（性别、居住地类型）*/
        .stRadio > div[role="radiogroup"] > label > div[data-testid="stMarkdownContainer"] {
            color: #333 !important;
        }
        
        /* Radio按钮的外圈和内部圆点 */
        .stRadio > div[role="radiogroup"] > label > div:first-child {
            border-color: #1a73e8 !important;
        }
        
        .stRadio > div[role="radiogroup"] > label:has(input:checked) > div:first-child {
            background-color: #1a73e8 !important;
            border-color: #1a73e8 !important;
        }
        
        .stRadio > div[role="radiogroup"] > label:has(input:checked) > div:first-child::after {
            background-color: white !important;
        }
        

        
        /* Number Input +/- 按钮颜色 - 蓝色（ADL总分、IADL总分）*/
        div[data-testid="stNumberInput"] button {
            color: #1a73e8 !important;
            border-color: #1a73e8 !important;
        }
        
        div[data-testid="stNumberInput"] button:hover {
            background-color: #e3f2fd !important;
            color: #1557b0 !important;
            border-color: #1557b0 !important;
        }
        
        /* ADL和IADL应用按钮样式 - 蓝色 */
        .adl-apply-button button, .iadl-apply-button button {
            background-color: #1a73e8 !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 2px 4px rgba(26, 115, 232, 0.2) !important;
        }
        
        .adl-apply-button button:hover, .iadl-apply-button button:hover {
            background-color: #1557b0 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(26, 115, 232, 0.3) !important;
        }
        
        /* 风险等级样式 */
        .risk-high {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .risk-medium {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .risk-low {
            background-color: #d1ecf1;
            border-left: 5px solid #17a2b8;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* 侧边栏样式 */
        .css-1d391kg {
            padding-top: 2rem;
        }
        
        /* 侧边栏系统信息专区样式 */
        [data-testid="stSidebar"] .system-info {
            margin-top: -1rem;
        }

        [data-testid="stSidebar"] .system-info h3 {
            font-size: 18px !important;      /* 与快速体验标题保持一致 */
            line-height: 1.4 !important;
            margin-bottom: 0.5rem;
            color: #1a73e8;
        }

        [data-testid="stSidebar"] .system-info p {
            font-size: 14.5px !important;      /* 内容字号 */
            line-height: 1.6 !important;
            margin: 2px 0 !important;
            color: #555;
        }

        [data-testid="stSidebar"] .system-info .version {
            margin-top: 0.5rem;
            color: #888;
        }

        /* 响应式调整 */
        @media (max-width: 768px) {
            .stColumn {
                width: 100% !important;
            }
        }
        </style>
        ''', unsafe_allow_html=True)
        
        # 添加心跳图标
        st.markdown('<i class="fas fa-heartbeat" style="font-size: 3.5rem; color: #1a73e8; margin-bottom: 1rem; display: block; text-align: center;"></i>', unsafe_allow_html=True)
        
        st.title("抑郁风险预警系统")

        # 导航 - 使用自定义按钮实现
        nav_items = [
            {"id": "assessment", "label": "🔍 风险评估", "icon": "🔍"},
            {"id": "explanation", "label": "🧠 模型解释", "icon": "🧠"},
            {"id": "instructions", "label": "ℹ️ 使用说明", "icon": "ℹ️"}
        ]
        
        # 在本地部署模式下添加历史记录选项
        if DEPLOY_MODE == "local":
            nav_items.insert(1, {"id": "history", "label": "📊 历史记录", "icon": "📊"})
        
        # 检查当前页面
        if "page" not in st.session_state:
            st.session_state.page = "🔍 风险评估"
        
        # 渲染导航菜单
        nav_container = st.container()
        with nav_container:
            for item in nav_items:
                is_active = st.session_state.page == item["label"]
                # 使用st.button实现导航项
                if st.button(
                    item["label"],
                    key=f"nav_{item['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.page = item["label"]
                    st.rerun()
        
        # 设置页面变量
        page = st.session_state.page

        st.divider()

        # 示例数据 - 使用与系统信息相同的样式
        st.markdown('''
        <div class="system-info">
            <h3>💡 快速体验</h3>
        </div>
        ''', unsafe_allow_html=True)
        
        # 创建按钮（上下排列）
        example_container = st.container()
        with example_container:
            # 高风险示例按钮
            if st.button("⚠️ 高风险示例", use_container_width=True, key="high_risk_button"):
                load_example_data(high_risk=True)
                # 自动触发评估
                st.session_state.auto_evaluate = True
                st.rerun()
            
            # 低风险示例按钮
            if st.button("✅ 低风险示例", use_container_width=True, key="low_risk_button"):
                load_example_data(high_risk=False)
                # 自动触发评估
                st.session_state.auto_evaluate = True
                st.rerun()
            
            # 重置按钮
            if st.button("⟳ 重置默认值", type="secondary", use_container_width=True):
                reset_form()
                st.success("表单已重置！")
                st.rerun()

        st.divider()

        # 获取版本信息
        version_number = CONFIG.get('version', {}).get('number', '1.0.0')
        version_year = CONFIG.get('version', {}).get('year', '2026')
        
        # 系统信息 - 使用自定义HTML容器控制样式
        st.markdown(f'''
        <div class="system-info">
            <h3>📊 系统信息</h3>
            <div style="display: flex; align-items: center; gap: 10px; margin: 8px 0;">
                <i class="fas fa-flask" style="color: #1a73e8; font-size: 16px;"></i>
                <p style="margin: 0;">科学评估：基于CatBoost模型</p>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; margin: 8px 0;">
                <i class="fas fa-bolt" style="color: #1a73e8; font-size: 16px;"></i>
                <p style="margin: 0;">快速筛查：仅需2-3分钟</p>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; margin: 8px 0;">
                <i class="fas fa-chart-bar" style="color: #1a73e8; font-size: 16px;"></i>
                <p style="margin: 0;">解释透明：提供影响因素分析</p>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; margin: 8px 0;">
                <i class="fas fa-shield-alt" style="color: #1a73e8; font-size: 16px;"></i>
                <p style="margin: 0;">隐私保护：完全匿名，不存储个人信息</p>
            </div>
            <div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #e0e0e0;">
                <p class="version">版本 {version_number} | {version_year}</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    return page


def render_assessment_page():
    """渲染风险评估页面"""
    # 主标题
    st.markdown("""
    <div style="text-align: center; margin-bottom: 10px;">
        <h1 style="font-family: 'Microsoft YaHei'; font-size: 28px; color: #1a73e8; font-weight: bold; margin: 0;">
            👋 欢迎使用中老年抑郁风险预警系统
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # 副标题
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <p style="font-family: 'Microsoft YaHei'; font-size: 16px; color: #4285f4; margin: 0;">
            面向社区中老年人 | 简易健康评估 | 早期风险筛查
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 免责声明
    st.markdown("""
    <div style="background-color: #E3F2FD; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <p style="font-family: 'Microsoft YaHei'; font-size: 16px; color: #1565C0; line-height: 1.5;">
            <strong>免责声明：</strong>本系统仅为抑郁风险初步筛查工具，不替代专业医疗诊断。评估结果仅供参考，如有身体不适或情绪困扰，请及时前往社区卫生服务中心或心理科进一步检查。本系统不存储个人身份信息，所有数据均为匿名处理。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 单列布局
    render_input_form()
    
    # 显示预测结果
    if 'last_prediction' in st.session_state:
        render_prediction_result()
    else:
        render_welcome_message()


def render_input_form():
    """渲染输入表单"""
    st.markdown("### 📋 请填写以下信息")

    # 基本信息分组
    with st.container():
        st.markdown("""
        <div style="background-color: #f0f8ff; border-radius: 10px; padding: 10px; margin-bottom: 15px; border-left: 5px solid #1e88e5;">
            <div style="display: flex; align-items: center; gap: 10px; height: 24px;">
                <span style="font-size: 20px; display: inline-block; vertical-align: middle;">👤</span>
                <h3 style="margin: 0; color: #1565c0; font-size: 20px; display: inline-block; vertical-align: middle;">基本信息</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.form_data['age'] = st.number_input(
                "年龄（岁）",
                min_value=45,
                max_value=120,
                value=st.session_state.form_data.get('age', 65),
                help="请输入45-120岁之间的年龄"
            )

            st.session_state.form_data['gender'] = st.radio(
                "性别",
                options=["男", "女"],
                horizontal=True,
                index=0 if st.session_state.form_data.get('gender') == '男' else 1
            )

        with col2:
            st.session_state.form_data['education_level'] = st.selectbox(
                "最高教育程度",
                options=list(EDUCATION_MAP.keys()),
                index=list(EDUCATION_MAP.keys()).index(
                    st.session_state.form_data.get('education_level', '初中')
                )
            )

            st.session_state.form_data['residence_type'] = st.radio(
                "居住地类型",
                options=["城镇", "农村"],
                horizontal=True,
                index=0 if st.session_state.form_data.get('residence_type') == '城镇' else 1
            )

    # 健康状况分组
    with st.container():
        st.markdown("""
        <div style="background-color: #f0f8ff; border-radius: 10px; padding: 10px; margin-bottom: 15px; border-left: 5px solid #1e88e5;">
            <div style="display: flex; align-items: center; gap: 10px; height: 24px;">
                <span style="font-size: 20px; display: inline-block; vertical-align: middle;">💪</span>
                <h3 style="margin: 0; color: #1565c0; font-size: 20px; display: inline-block; vertical-align: middle;">健康状况</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.form_data['self_rated_health'] = st.selectbox(
                "自评健康状况",
                options=HEALTH_OPTIONS,
                index=HEALTH_OPTIONS.index(
                    st.session_state.form_data.get('self_rated_health', '好')
                )
            )

        with col2:
            st.session_state.form_data['childhood_health'] = st.selectbox(
                "儿童时期健康状况",
                options=HEALTH_OPTIONS_CHILDHOOD,
                index=HEALTH_OPTIONS_CHILDHOOD.index(
                    st.session_state.form_data.get('childhood_health', '好')
                )
            )

    # 功能状态分组
    with st.container():
        st.markdown("""
        <div style="background-color: #f0f8ff; border-radius: 10px; padding: 10px; margin-bottom: 15px; border-left: 5px solid #1e88e5;">
            <div style="display: flex; align-items: center; gap: 10px; height: 24px;">
                <span style="font-size: 20px; display: inline-block; vertical-align: middle;">🏃</span>
                <h3 style="margin: 0; color: #1565c0; font-size: 20px; display: inline-block; vertical-align: middle;">功能状态</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 添加CSS使两列内容底部对齐
        st.markdown("""
        <style>
        /* ADL和IADL行的列容器 */
        div[data-testid="column"] > div {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            height: 100% !important;
        }
        
        /* 确保popover按钮与输入框底部对齐 */
        .stPopover {
            margin-top: auto !important;
        }
        
        /* number_input和popover在同一行时底部对齐 */
        div[data-testid="stHorizontalBlock"] {
            align-items: flex-end !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            col_adl, col_button = st.columns([3, 1])
            with col_adl:
                st.session_state.form_data['ADL_total'] = st.number_input(
                    "ADL总分（0-90）",
                    min_value=0,
                    max_value=90,
                    value=st.session_state.form_data.get('ADL_total', 90),
                    help="日常生活活动能力总分"
                )
            with col_button:
                render_adl_popover()

        with col2:
            col_iadl, col_button = st.columns([3, 1])
            with col_iadl:
                st.session_state.form_data['IADL_total'] = st.number_input(
                    "IADL总分（0-6）",
                    min_value=0,
                    max_value=6,
                    value=st.session_state.form_data.get('IADL_total', 6),
                    help="工具性日常生活活动能力总分"
                )
            with col_button:
                render_iadl_popover()

    # 生活方式分组
    with st.container():
        st.markdown("""
        <div style="background-color: #f0f8ff; border-radius: 10px; padding: 10px; margin-bottom: 15px; border-left: 5px solid #1e88e5;">
            <div style="display: flex; align-items: center; gap: 10px; height: 24px;">
                <span style="font-size: 20px; display: inline-block; vertical-align: middle;">🌙</span>
                <h3 style="margin: 0; color: #1565c0; font-size: 20px; display: inline-block; vertical-align: middle;">生活方式</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.form_data['sleep_hours_night'] = st.slider(
            "夜间睡眠时长（小时）",
            min_value=0.0,
            max_value=12.0,
            value=float(st.session_state.form_data.get('sleep_hours_night', 7.0)),
            step=0.5,
            help="平均每晚睡眠时长"
        )

        st.markdown("疼痛部位（可多选）")
        # 全选/清空按钮
        col_select, col_clear = st.columns(2)
        with col_select:
            if st.button("全选", key="pain_select_all"):
                # 1. 更新业务数据
                st.session_state.form_data['pain_sites'] = PAIN_SITES.copy()
                # 2. 更新每个复选框的 Streamlit 内部状态
                for site in PAIN_SITES:
                    st.session_state[f"pain_{site}"] = True
                st.rerun()
        with col_clear:
            if st.button("清空", key="pain_clear_all"):
                st.session_state.form_data['pain_sites'] = []
                for site in PAIN_SITES:
                    st.session_state[f"pain_{site}"] = False
                st.rerun()
        
        # 网格布局，每行4个
        num_columns = 4
        pain_sites = PAIN_SITES
        for i in range(0, len(pain_sites), num_columns):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i + j < len(pain_sites):
                    site = pain_sites[i + j]
                    # 直接用 key 控制，无需手动维护 form_data（但为了业务一致性，仍需同步）
                    checked = cols[j].checkbox(site, key=f"pain_{site}")
                    # 同步业务数据：当复选框状态变化时，更新 form_data['pain_sites']
                    if checked and site not in st.session_state.form_data['pain_sites']:
                        st.session_state.form_data['pain_sites'].append(site)
                    elif not checked and site in st.session_state.form_data['pain_sites']:
                        st.session_state.form_data['pain_sites'].remove(site)
        st.caption(f"已选择 {len(st.session_state.form_data.get('pain_sites', []))} 个部位")

    # 慢性病管理分组
    with st.container():
        st.markdown("""
        <div style="background-color: #e3f2fd; border-radius: 10px; padding: 10px; margin-bottom: 15px; border-left: 5px solid #2196f3;">
            <div style="display: flex; align-items: center; gap: 10px; height: 24px;">
                <span style="font-size: 20px; display: inline-block; vertical-align: middle;">🏥</span>
                <h3 style="margin: 0; color: #1565c0; font-size: 20px; display: inline-block; vertical-align: middle;">慢性病管理</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # 慢性病选择
        st.markdown("患有以下慢性病（可多选）")
        # 全选/清空按钮
        col_select, col_clear = st.columns(2)
        with col_select:
            if st.button("全选", key="chronic_select_all"):
                st.session_state.form_data['chronic_diseases'] = CHRONIC_DISEASES.copy()
                for disease in CHRONIC_DISEASES:
                    st.session_state[f"chronic_{disease}"] = True
                # 更新共病簇
                stomach_arthritis, arthritis_asthma = update_comorbidities(st.session_state.form_data['chronic_diseases'])
                st.session_state.form_data['stomach_arthritis_pair'] = stomach_arthritis
                st.session_state.form_data['arthritis_asthma_pair'] = arthritis_asthma
                st.rerun()
        with col_clear:
            if st.button("清空", key="chronic_clear_all"):
                st.session_state.form_data['chronic_diseases'] = []
                for disease in CHRONIC_DISEASES:
                    st.session_state[f"chronic_{disease}"] = False
                st.session_state.form_data['stomach_arthritis_pair'] = False
                st.session_state.form_data['arthritis_asthma_pair'] = False
                st.rerun()
        
        # 网格布局，每行4个
        num_columns = 4
        chronic_diseases = CHRONIC_DISEASES
        for i in range(0, len(chronic_diseases), num_columns):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i + j < len(chronic_diseases):
                    disease = chronic_diseases[i + j]
                    checked = cols[j].checkbox(disease, key=f"chronic_{disease}")
                    # 同步业务数据
                    if checked and disease not in st.session_state.form_data['chronic_diseases']:
                        st.session_state.form_data['chronic_diseases'].append(disease)
                        # 实时更新共病簇
                        stomach_arthritis, arthritis_asthma = update_comorbidities(st.session_state.form_data['chronic_diseases'])
                        st.session_state.form_data['stomach_arthritis_pair'] = stomach_arthritis
                        st.session_state.form_data['arthritis_asthma_pair'] = arthritis_asthma
                    elif not checked and disease in st.session_state.form_data['chronic_diseases']:
                        st.session_state.form_data['chronic_diseases'].remove(disease)
                        stomach_arthritis, arthritis_asthma = update_comorbidities(st.session_state.form_data['chronic_diseases'])
                        st.session_state.form_data['stomach_arthritis_pair'] = stomach_arthritis
                        st.session_state.form_data['arthritis_asthma_pair'] = arthritis_asthma
        st.caption(f"已选择 {len(st.session_state.form_data.get('chronic_diseases', []))} 种慢性病")

        # 显示共病簇提示
        selected_diseases = st.session_state.form_data.get('chronic_diseases', [])
        stomach_arthritis, arthritis_asthma = update_comorbidities(selected_diseases)
        if stomach_arthritis:
            st.success("🔍 检测到：胃病-关节炎共病")
        if arthritis_asthma:
            st.success("🔍 检测到：关节炎-哮喘共病")



    # 表单提交按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit = st.button(
            "🚀 开始评估",
            type="primary",
            use_container_width=True
        )

    if submit:
        # 验证输入
        errors = validate_input(st.session_state.form_data)
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            # 根据慢性病选择计算共病簇
            selected_diseases = st.session_state.form_data.get('chronic_diseases', [])
            stomach_arthritis, arthritis_asthma = update_comorbidities(selected_diseases)
            st.session_state.form_data['stomach_arthritis_pair'] = stomach_arthritis
            st.session_state.form_data['arthritis_asthma_pair'] = arthritis_asthma

            # 进行预测
            with st.spinner("正在评估中，请稍候..."):
                result = perform_prediction(st.session_state.form_data)
                if result:
                    st.session_state.last_prediction = result
                    st.session_state.show_result = True
                    st.rerun()


def render_prediction_result():
    """渲染预测结果"""
    if not st.session_state.get('show_result', False):
        return

    result = st.session_state.last_prediction
    risk_info = get_risk_info(result['probability'])

    # 主标题
    st.markdown("## 📊 评估结果")

    # 风险概览卡片 - 左侧仪表盘，右侧文字
    with st.container():
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 10px; margin-bottom: 15px; border-left: 5px solid {risk_info['color']};">
            <div style="display: flex; align-items: center; gap: 10px; height: 24px;">
                <span style="font-size: 20px; display: inline-block; vertical-align: middle;">🎯</span>
                <h3 style="margin: 0; color: #333; font-size: 20px; display: inline-block; vertical-align: middle;">风险概览</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 生成风险仪表盘
        import plotly.graph_objects as go
        
        probability = result['probability']
        # 从配置中获取概率阈值和颜色
        risk_config = CONFIG.get('risk', {})
        thresholds = risk_config.get('probability_thresholds', [0.3, 0.6])
        colors = risk_config.get('probability_colors', ['#4caf50', '#ff9800', '#f44336'])
        
        # 根据概率确定颜色
        if probability < thresholds[0]:
            color = colors[0]      # 绿色
        elif probability < thresholds[1]:
            color = colors[1]      # 橙色
        else:
            color = colors[2]      # 红色
        
        # 创建圆环图（仪表盘）
        fig = go.Figure(
            go.Pie(
                values=[probability, 1 - probability],
                marker_colors=[color, '#e0e0e0'],
                hole=0.7,
                rotation=90,
                direction='clockwise',
                showlegend=False,
                textinfo='none'
            )
        )
        fig.add_annotation(
            text=f"{probability:.0%}",
            x=0.5, y=0.5,
            font=dict(size=24),
            showarrow=False
        )
        fig.update_layout(
            width=200, height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # 使用两列布局，确保高度一致
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; justify-content: center; height: 200px;">
                <div style="margin-bottom: 10px;">
                    <p style="margin: 0 0 5px 0; color: #666; font-size: 14px;">抑郁风险概率</p>
                    <p style="margin: 0; font-size: 28px; font-weight: bold; color: #333;">{probability:.3f}</p>
                </div>
                <div>
                    <p style="margin: 0 0 5px 0; color: #666; font-size: 14px;">风险类别</p>
                    <span style="padding: 5px 15px; border-radius: 20px; background-color: {risk_info['color']}; color: white; font-weight: bold;">
                        {risk_info['risk_category']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 主要影响因素分析卡片
    with st.container():
        st.markdown("""
        <div style="background-color: #e3f2fd; border-radius: 10px; padding: 10px; margin-bottom: 15px; border-left: 5px solid #2196f3;">
            <div style="display: flex; align-items: center; gap: 10px; height: 24px;">
                <span style="font-size: 20px; display: inline-block; vertical-align: middle;">📈</span>
                <h3 style="margin: 0; color: #1565c0; font-size: 20px; display: inline-block; vertical-align: middle;">主要影响因素分析</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if result.get('shap_values') is not None:
            # 生成交互式SHAP瀑布图
            import shap
            import numpy as np
            import tempfile
            import os
            import streamlit.components.v1 as components

            # 确保shap_values是一维数组
            shap_values = result['shap_values']
            if len(shap_values.shape) > 1:
                shap_values = shap_values.flatten()

            # 将英文特征名称转换为中文
            chinese_feature_names = [FEATURE_NAMES_CN.get(feat, feat) for feat in result['feature_names']]

            # 创建Explanation对象
            exp = shap.Explanation(
                values=shap_values,
                base_values=result['base_value'],
                data=np.zeros_like(shap_values),
                feature_names=chinese_feature_names
            )

            # 生成交互式SHAP瀑布图
            try:
                # 创建交互式瀑布图
                fig = create_interactive_shap_waterfall(
                    shap_values,
                    result['feature_names'],
                    result['base_value'],
                    result['probability'],
                    max_display=13
                )
                
                # 显示图表
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"无法生成交互式SHAP图: {str(e)}")
                # 回退到静态图表
                import matplotlib.pyplot as plt
                # 使用全局字体设置
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                
                # 创建图表
                plt.figure(figsize=(10, 8))
                shap.plots.waterfall(exp, max_display=13)
                
                # 显示图表
                st.pyplot(plt.gcf())
                plt.close()

            # 文字解释
            explanation = generate_text_explanation(
                shap_values,
                result['feature_names'],
                result['base_value'],
                result['probability']
            )
            if explanation:
                with st.container():
                    st.markdown("""
                    <div style="background-color: #f5f5f5; border-radius: 10px; padding: 10px; margin-top: 15px; margin-bottom: 15px; border-left: 5px solid #9e9e9e;">
                        <div style="display: flex; align-items: center; gap: 10px; height: 24px;">
                            <span style="font-size: 20px; display: inline-block; vertical-align: middle;">📋</span>
                            <h3 style="margin: 0; color: #333; font-size: 20px; display: inline-block; vertical-align: middle;">详细解释</h3>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(explanation)

    # 个性化建议卡片
    with st.container():
        st.markdown("""
        <div style="background-color: #f0fff4; border-radius: 10px; padding: 10px; margin-bottom: 15px; border-left: 5px solid #4caf50;">
            <div style="display: flex; align-items: center; gap: 10px; height: 24px;">
                <span style="font-size: 20px; display: inline-block; vertical-align: middle;">💡</span>
                <h3 style="margin: 0; color: #2e7d32; font-size: 20px; display: inline-block; vertical-align: middle;">个性化建议</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        advice = get_personalized_advice(
            risk_info,
            result.get('shap_values'),
            result.get('feature_names', []),
            result.get('feature_values', {})
        )
        st.markdown(advice)

    # 保存和分享选项
    st.divider()
    with st.container():
        st.markdown("""
        <div style="background-color: #f3e5f5; border-radius: 10px; padding: 10px; margin-bottom: 15px; border-left: 5px solid #9c27b0;">
            <div style="display: flex; align-items: center; gap: 10px; height: 24px;">
                <span style="font-size: 20px; display: inline-block; vertical-align: middle;">📤</span>
                <h3 style="margin: 0; color: #6a1b9a; font-size: 20px; display: inline-block; vertical-align: middle;">保存结果</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 根据部署模式调整布局
    if DEPLOY_MODE == "local":
        # 本地部署：显示备注、保存结果和重新评估按钮
        col1, col2, col3 = st.columns(3)

        with col1:
            # 添加备注
            nickname = st.text_input(
                "添加备注（可选）",
                placeholder="如：张阿姨_评估",
                max_chars=20,
                key="result_nickname"
            )

        with col2:
            if st.button("💾 保存结果", use_container_width=True):
                if save_prediction_to_db(result, nickname):
                    st.success("✅ 结果已保存！")
                else:
                    st.error("保存失败，请重试")

        with col3:
            if st.button("🔄 重新评估", use_container_width=True, type="secondary"):
                st.session_state.show_result = False
                st.rerun()
    else:
        # 云部署：显示PDF下载和重新评估按钮
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 下载PDF报告", use_container_width=True):
                # 生成PDF报告
                record_dict = {
                    'id': 'current',
                    'timestamp': result['timestamp'],
                    'nickname': '',
                    'probability': result['probability'],
                    'risk_level': '有风险' if result['is_risk'] else '无风险',
                    'input_features': json.dumps(result['feature_values'], ensure_ascii=False),
                    'shap_summary': None
                }
                with st.spinner("正在生成PDF报告..."):
                    pdf_data = generate_pdf_report(record_dict)
                    if pdf_data:
                        timestamp_str = result['timestamp'].replace(':', '-').replace(' ', '_')
                        file_name = f"depression_risk_report_{timestamp_str}.pdf"
                        st.download_button(
                            label="下载PDF",
                            data=pdf_data,
                            file_name=file_name,
                            mime="application/pdf",
                            key="download_pdf",
                            help="下载PDF报告",
                            use_container_width=True
                        )
                    else:
                        st.error("PDF生成失败")

        with col2:
            if st.button("🔄 重新评估", use_container_width=True, type="secondary"):
                st.session_state.show_result = False
                st.rerun()


def render_welcome_message():
    """渲染欢迎信息"""
    pass




def render_history_page():
    """渲染历史记录页面 - 表格形式，每行带删除和PDF按钮"""
    # 检查部署模式
    if DEPLOY_MODE != "local":
        st.error("历史记录功能仅在本地部署模式下可用")
        return
    
    # 密码验证
    if "history_access" not in st.session_state:
        st.session_state.history_access = False
    if "last_activity_time" not in st.session_state:
        st.session_state.last_activity_time = time.time()
    
    # 检查是否超时（10分钟）
    timeout = 10 * 60  # 10分钟，单位为秒
    current_time = time.time()
    if st.session_state.history_access and (current_time - st.session_state.last_activity_time) > timeout:
        del st.session_state.history_access
        st.session_state.last_activity_time = current_time
        st.rerun()
    
    if not st.session_state.history_access:
        # 修改标题为蓝色
        st.markdown("<h1 style='color: #1a73e8;'>🔒 历史记录访问</h1>", unsafe_allow_html=True)
        # 增大输入框和标签字体
        st.markdown("<style>input[type='password'] { font-size: 18px; } .stTextInput label { font-size: 18px; font-weight: 500; }</style>", unsafe_allow_html=True)
        password = st.text_input("请输入访问密码", type="password")
        # 修改按钮为蓝色
        if st.button("验证密码", type="primary"):
            if verify_admin_password(password):
                st.session_state.history_access = True
                st.session_state.last_activity_time = time.time()
                st.rerun()
            else:
                st.error("密码错误，请重试")
        return
    
    # 更新活动时间
    st.session_state.last_activity_time = time.time()
    
    st.title("📊 评估历史记录")

    # 初始化数据库
    init_db()
    
    # 密码修改功能
    with st.expander("🔧 修改密码"):
        current_password = st.text_input("当前密码", type="password")
        new_password = st.text_input("新密码", type="password")
        confirm_password = st.text_input("确认新密码", type="password")
        if st.button("更新密码"):
            if new_password != confirm_password:
                st.error("两次输入的新密码不一致")
            else:
                success, message = update_admin_password(current_password, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    # 筛选选项
    col1, col2, col3 = st.columns(3)
    with col1:
        time_filter = st.selectbox("时间范围", ["全部", "近7天", "近30天", "近3个月"])
    with col2:
        risk_filter = st.selectbox("风险等级", ["全部", "有风险", "无风险"])
    with col3:
        nickname_search = st.text_input("搜索备注", placeholder="输入备注关键词...")

    # 分页设置
    col4, col5 = st.columns([1, 1])
    with col4:
        page_size = st.selectbox("每页显示", [10, 20, 50], index=1)
    
    # 获取历史记录和总记录数（单次查询）
    offset = 0
    page = 1
    records_df, total_count = get_history(time_filter, risk_filter, nickname_search, limit=page_size, offset=offset)
    
    with col5:
        total_pages = (total_count + page_size - 1) // page_size
        page = st.number_input("页码", min_value=1, max_value=total_pages if total_pages > 0 else 1, value=1)
        # 重新计算偏移量
        offset = (page - 1) * page_size
        # 如果页码变化，重新获取数据
        if offset > 0:
            records_df, total_count = get_history(time_filter, risk_filter, nickname_search, limit=page_size, offset=offset)

    if records_df.empty:
        st.info("暂无历史记录")
        if st.button("返回评估页面", use_container_width=True):
            st.session_state.page = "🔍 风险评估"
            st.rerun()
        return

    st.markdown(f"### 共找到 {total_count} 条记录，显示第 {offset + 1}-{min(offset + page_size, total_count)} 条")
    st.markdown(f"第 {page} 页，共 {total_pages} 页")

    # 存储选中的记录ID
    if "selected_ids" not in st.session_state:
        st.session_state.selected_ids = []

    # 表头
    header_cols = st.columns([0.5, 1, 2.5, 1.5, 1.5, 1.5, 1.5])
    # 全选复选框
    all_selected = len(st.session_state.selected_ids) == len(records_df)
    select_all = header_cols[0].checkbox("全选", value=all_selected, key="select_all")
    
    # 处理全选/取消全选 - 直接设置每个单选框的 session_state
    if select_all and not all_selected:
        # 全选：将所有单选框的 session_state 设为 True
        for _, row in records_df.iterrows():
            st.session_state[f"select_{row['id']}"] = True
        st.session_state.selected_ids = records_df['id'].tolist()
        st.rerun()
    elif not select_all and all_selected:
        # 取消全选：将所有单选框的 session_state 设为 False
        for _, row in records_df.iterrows():
            st.session_state[f"select_{row['id']}"] = False
        st.session_state.selected_ids = []
        st.rerun()
    
    header_cols[1].write("ID")
    header_cols[2].write("时间")
    header_cols[3].write("备注")
    header_cols[4].write("风险概率")
    header_cols[5].write("风险类别")
    header_cols[6].write("操作")
    st.divider()

    # 遍历显示每条记录
    for idx, row in records_df.iterrows():
        record_id = row['id']

        cols = st.columns([0.5, 1, 2.5, 1.5, 1.5, 1.5, 1.5])
        # 复选框 - 只传 key，让 Streamlit 自动从 session_state 中读取状态
        checked = cols[0].checkbox("", key=f"select_{record_id}")

        cols[1].write(record_id)
        cols[2].write(row['timestamp'])
        cols[3].write(row['nickname'] if row['nickname'] else "无")
        cols[4].write(f"{row['probability']:.3f}")
        risk_color = "red" if row['risk_level'] == "有风险" else "green"
        cols[5].markdown(f"<span style='color:{risk_color}; font-weight:bold;'>{row['risk_level']}</span>", unsafe_allow_html=True)

        # 操作按钮（删除和PDF） - 改为 emoji + 文字，并固定在每行最右侧
        with cols[6]:
            # 使用两列布局让按钮并排
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🗑️ 删除", key=f"del_{record_id}", help="删除记录", use_container_width=True):
                    delete_record(record_id)
                    if record_id in st.session_state.selected_ids:
                        st.session_state.selected_ids.remove(record_id)
                    st.success("记录已删除")
                    st.rerun()
            with btn_col2:
                # 检查会话状态中是否有PDF数据
                if f"pdf_data_{record_id}" in st.session_state:
                    # 如果有PDF数据，直接显示下载按钮
                    pdf_data = st.session_state[f"pdf_data_{record_id}"]
                    file_name = st.session_state[f"pdf_file_name_{record_id}"]
                    
                    # 显示下载按钮
                    st.download_button(
                        label="下载PDF",
                        data=pdf_data,
                        file_name=file_name,
                        mime="application/pdf",
                        key=f"download_{record_id}",
                        help="下载PDF报告",
                        use_container_width=True
                    )
                    
                    # 清理会话状态
                    del st.session_state[f"pdf_data_{record_id}"]
                    del st.session_state[f"pdf_file_name_{record_id}"]
                else:
                    # 如果没有PDF数据，显示生成按钮
                    if st.button("📄 PDF", key=f"pdf_{record_id}", help="导出PDF报告", use_container_width=True):
                        record_dict = row.to_dict()
                        with st.spinner("正在生成PDF报告..."):
                            pdf_data = generate_pdf_report(record_dict)
                            if pdf_data:
                                timestamp_str = str(row['timestamp'])
                                safe_timestamp = timestamp_str.replace(':', '-').replace(' ', '_')
                                file_name = f"report_{safe_timestamp}.pdf"
                                
                                # 存储PDF数据到会话状态
                                st.session_state[f"pdf_data_{record_id}"] = pdf_data
                                st.session_state[f"pdf_file_name_{record_id}"] = file_name
                                
                                # 重新运行以显示下载按钮
                                st.rerun()
                            else:
                                st.error("PDF生成失败")

    # 在渲染循环之后，根据所有单选框的实际状态重建 selected_ids
    new_selected = []
    for _, row in records_df.iterrows():
        if st.session_state.get(f"select_{row['id']}", False):
            new_selected.append(row['id'])
    st.session_state.selected_ids = new_selected

    # 批量操作区域（已删除清除选择按钮）
    st.divider()
    selected_ids = st.session_state.selected_ids
    selected_count = len(selected_ids)
    if selected_count > 0:
        st.info(f"已选择 {selected_count} 条记录")
        # 改为三列布局（批量删除、导出CSV、批量导出PDF）
        col1, col2, col3 = st.columns(3)
        with col1:
            # 批量删除按钮（文字改为"批量删除"）
            if st.button("批量删除", use_container_width=True):
                for rid in selected_ids:
                    delete_record(rid)
                st.session_state.selected_ids.clear()
                st.success(f"已删除 {selected_count} 条记录")
                st.rerun()
        with col2:
            # 导出CSV按钮（文字改为"导出CSV"，点击时才生成文件）
            if st.button("导出CSV", use_container_width=True, key="btn_export_csv"):
                selected_records = records_df[records_df['id'].isin(selected_ids)]
                with st.spinner("正在生成CSV文件..."):
                    csv_data = export_to_csv(selected_records)
                    if csv_data:
                        st.download_button(
                            label="下载CSV文件",
                            data=csv_data,
                            file_name=f"selected_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="batch_csv_download",
                            use_container_width=True
                        )
        with col3:
            # 批量导出PDF按钮（文字改为"批量导出PDF"，点击时才生成文件）
            if st.button("批量导出PDF", use_container_width=True, key="btn_export_pdf"):
                selected_records = records_df[records_df['id'].isin(selected_ids)]
                with st.spinner("正在生成PDF报告包..."):
                    zip_data = batch_export_pdf(selected_records)
                    if zip_data:
                        st.download_button(
                            label="下载PDF包",
                            data=zip_data,
                            file_name=f"reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            key="batch_pdf_download",
                            use_container_width=True
                        )
    else:
        st.info("未选择任何记录，请勾选记录后批量操作")

    # 统计信息
    st.divider()
    stats = get_statistics()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总记录数", stats['total'])
    with col2:
        st.metric("有风险记录", stats['risk_count'])
    with col3:
        st.metric("平均风险概率", f"{stats['avg_probability']:.3f}")


@st.cache_data

def load_shap_data():
    """加载SHAP分析数据（缓存）"""
    import os
    import pandas as pd
    
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 从配置文件获取SHAP分析结果路径
    shap_config = CONFIG.get('shap', {})
    local_shap_dir = os.path.join(current_dir, shap_config.get('local_shap_dir', 'shap_results'))
    original_shap_dir = shap_config.get('original_shap_dir', r"D:\AAA毕业\AAA毕业设计\3Ending\主实验-二分类\机器学习-二分类\output\evaluation\shap")
    
    # 选择使用哪个路径
    if os.path.exists(local_shap_dir) and len(os.listdir(local_shap_dir)) > 0:
        shap_dir = local_shap_dir
    else:
        shap_dir = original_shap_dir
    
    data = {
        'shap_dir': shap_dir,
        'shap_importance': None,
        'shap_df': None,
        'X_sample': None,
        'feature_names': None
    }
    
    # 加载特征重要性数据
    feature_importance_path = os.path.join(shap_dir, "shap_feature_importance.csv")
    if os.path.exists(feature_importance_path):
        data['shap_importance'] = pd.read_csv(feature_importance_path)
    
    # 加载SHAP值数据
    shap_values_path = os.path.join(shap_dir, "shap_values.csv")
    if os.path.exists(shap_values_path):
        data['shap_df'] = pd.read_csv(shap_values_path)
        data['feature_names'] = [col for col in data['shap_df'].columns if col not in ['predicted_probability', 'sample_id']]
        
        # 加载原始测试数据
        local_test_data_path = os.path.join(current_dir, shap_config.get('local_test_data_path', 'shap_results/X_test.csv'))
        original_test_data_path = shap_config.get('original_test_data_path', r"D:\AAA毕业\AAA毕业设计\3Ending\主实验-二分类\机器学习-二分类\output\data\X_test.csv")
        
        if os.path.exists(local_test_data_path):
            test_data_path = local_test_data_path
        else:
            test_data_path = original_test_data_path
        
        if os.path.exists(test_data_path):
            X_sample = pd.read_csv(test_data_path)
            
            # 确保特征顺序一致
            if data['feature_names']:
                common_features = [f for f in data['feature_names'] if f in X_sample.columns]
                if len(common_features) != len(data['feature_names']):
                    data['feature_names'] = common_features
                
                X_sample = X_sample[common_features]
                
                # 确保数据形状匹配
                if data['shap_df'] is not None and data['feature_names']:
                    shap_values = data['shap_df'][data['feature_names']].values
                    if shap_values.shape[0] != X_sample.shape[0]:
                        # 直接使用前N个样本
                        min_samples = min(shap_values.shape[0], X_sample.shape[0])
                        data['shap_df'] = data['shap_df'].iloc[:min_samples]
                        X_sample = X_sample.iloc[:min_samples]
                
                data['X_sample'] = X_sample
    
    return data


def render_global_explanation_page():
    """渲染全局SHAP解释页面"""
    st.title("模型全局解释")
    
    # 页面说明
    st.markdown("""
    <div style="background-color: #e3f2fd; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
        <h4 style="margin: 0 0 10px 0; color: #1565c0;">关于模型解释</h4>
        <p style="margin: 0; color: #333; line-height: 1.5;">
            本页面展示了模型的全局解释，帮助您理解哪些因素对抑郁风险预测影响最大。
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("正在加载SHAP分析结果..."):
        try:
            import os
            import matplotlib.pyplot as plt
            import plotly.express as px
            
            # 使用全局字体设置
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            # 加载SHAP数据（使用缓存）
            data = load_shap_data()
            shap_dir = data['shap_dir']
            
            # 检查SHAP分析结果目录是否存在
            if not os.path.exists(shap_dir):
                st.error(f"SHAP分析结果目录不存在: {shap_dir}")
                st.info("请先运行 05shap_analysis.py 生成分析结果")
                return
            
            # 显示特征重要性
            if data['shap_importance'] is not None:
                st.markdown("## 🌍 全局特征重要性")
                
                # 显示特征重要性表格
                st.dataframe(data['shap_importance'][['feature_cn', 'shap_importance']].rename(columns={'feature_cn': '特征名称', 'shap_importance': 'SHAP重要性'}), use_container_width=True)
                
                # 使用Plotly创建交互式条形图
                fig = px.bar(
                    data['shap_importance'],
                    y='feature_cn',
                    x='shap_importance',
                    orientation='h',
                    title='全局特征重要性',
                    labels={'feature_cn': '特征名称', 'shap_importance': '平均 |SHAP 值|'},
                    color='shap_importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=600,
                    yaxis={'categoryorder': 'total ascending'},
                    margin=dict(l=100, r=50, t=80, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("特征重要性文件不存在")
            
            # 特征依赖图
            if data['shap_df'] is not None and data['X_sample'] is not None and data['feature_names']:
                st.markdown("## 📈 特征依赖分析")
                st.info("展示每个特征与SHAP值的关系，帮助理解特征值如何影响抑郁风险预测")
                
                feature_names_cn = get_feature_names_cn()
                selected_feature = st.selectbox(
                    "选择要分析的特征",
                    options=data['feature_names'],
                    format_func=lambda x: feature_names_cn.get(x, x),
                    help="选择一个特征查看其与SHAP值的关系"
                )
                
                # 尝试加载预先生成的依赖图
                import os
                shap_dir = data['shap_dir']
                dependence_image_path = os.path.join(shap_dir, f"shap_dependence_{selected_feature}.png")
                
                if os.path.exists(dependence_image_path):
                    # 直接显示预先生成的图片（图片上已自带标题）
                    st.image(dependence_image_path)
                else:
                    # 如果预先生成的图片不存在，回退到动态生成
                    st.warning("预先生成的依赖图不存在，正在动态生成...")
                    # 提取SHAP值
                    shap_values = data['shap_df'][data['feature_names']].values
                    
                    # 生成依赖图
                    from utils.shap_utils import generate_dependence_plot
                    fig_dependence = generate_dependence_plot(
                        shap_values, data['X_sample'], selected_feature, feature_names_cn
                    )
                    if fig_dependence:
                        st.pyplot(fig_dependence)
                    else:
                        st.warning("无法生成依赖图，请检查数据格式")
            else:
                st.warning("SHAP值文件或测试数据文件不存在")
            
            # 模型信息
            st.markdown("## 🛠️ 模型技术原理")
            st.markdown("""
            - **模型类型**: CatBoost分类器（梯度提升决策树）
            - **特征数量**: 13个健康指标
            - **解释方法**: SHAP (SHapley Additive exPlanations)
            - **分析样本**: 基于测试集数据
            - **训练数据**: 中国健康与养老追踪调查(CHARLS)数据
            """)
            
            # 解释说明
            st.markdown("## 📖 解释说明")
            st.markdown("""
            - **SHAP值**: 表示每个特征对预测结果的贡献程度
            - **正SHAP值**: 增加抑郁风险的因素
            - **负SHAP值**: 降低抑郁风险的因素
            - **全局特征重要性**: 基于所有样本的平均SHAP绝对值计算
            - **特征依赖图**: 展示特征值如何影响SHAP值
            """)
            
        except Exception as e:
            st.error(f"加载SHAP分析结果时发生错误: {str(e)}")
            # 显示详细错误信息
            st.code(str(e))
            st.info("请确保SHAP分析结果文件存在且格式正确")


def render_instructions_page():
    """渲染使用说明页面 - 现代化设计"""
    st.title("ℹ️ 使用说明")

    # 系统简介（现代化渐变卡片）
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e88e5 0%, #64b5f6 100%); padding: 25px; border-radius: 16px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(30, 136, 229, 0.2);">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background-color: rgba(255, 255, 255, 0.2); border-radius: 50%; padding: 12px; margin-right: 15px;">
                <i class="fas fa-brain" style="font-size: 24px; color: white;"></i>
            </div>
            <h2 style="color: white; margin: 0; font-size: 24px;">系统简介</h2>
        </div>
        <p style="color: white; margin: 0; line-height: 1.7; font-size: 16px;">
            本系统基于 <strong>中国健康与养老追踪调查(CHARLS)</strong> 数据构建的 CatBoost 机器学习模型，
            专为社区中老年人设计。通过 <strong>13项简易健康指标</strong>，快速评估抑郁风险，并提供个性化建议。
        </p>
        <div style="margin-top: 15px; display: flex; gap: 10px;">
            <span style="background-color: rgba(255, 255, 255, 0.2); color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px;">科学评估</span>
            <span style="background-color: rgba(255, 255, 255, 0.2); color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px;">快速筛查</span>
            <span style="background-color: rgba(255, 255, 255, 0.2); color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px;">个性化建议</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 操作步骤（现代化步骤指示器）
    st.markdown("## 📝 使用步骤")
    steps = [
        {
            "icon": "👤",
            "title": "填写健康信息",
            "content": "根据页面提示填写基本信息、健康状况、功能状态、生活方式和慢性病信息"
        },
        {
            "icon": "🚀",
            "title": "开始评估",
            "content": "点击「开始评估」按钮，系统将分析您的健康数据"
        },
        {
            "icon": "📊",
            "title": "查看结果",
            "content": "查看风险评估结果、主要影响因素分析和个性化建议"
        },
        {
            "icon": "💾",
            "title": "保存记录",
            "content": "可选择保存评估结果到历史记录，方便后续查看和对比"
        }
    ]

    # 步骤卡片
    step_container = st.container()
    with step_container:
        for i, step in enumerate(steps, 1):
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start; background-color: #f8f9fa; border-radius: 12px; padding: 20px; margin-bottom: 15px; border: 1px solid #e0e0e0; transition: all 0.3s ease; hover: box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="background-color: #1e88e5; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: bold; margin-right: 15px; flex-shrink: 0;">
                    {i}
                </div>
                <div style="flex: 1;">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 20px; margin-right: 10px;">{step['icon']}</span>
                        <h3 style="margin: 0; font-size: 18px; color: #333;">{step['title']}</h3>
                    </div>
                    <p style="margin: 0; color: #666; line-height: 1.5;">{step['content']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 详细填写指南（可折叠部分）
    st.markdown("## 📋 详细填写指南")
    guide_container = st.container()
    with guide_container:
        with st.expander("👤 基本信息", expanded=True):
            st.markdown("""
            <div style="background-color: #f0f8ff; border-radius: 10px; padding: 15px;">
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>年龄</strong>：请输入45-120岁之间的实际年龄</li>
                    <li><strong>性别</strong>：选择「男」或「女」</li>
                    <li><strong>教育程度</strong>：选择您的最高教育水平，系统会自动转换为受教育年限</li>
                    <li><strong>居住地</strong>：选择「城镇」或「农村」</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("💪 健康状况", expanded=False):
            st.markdown("""
            <div style="background-color: #f0fff4; border-radius: 10px; padding: 15px;">
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>自评健康</strong>：根据您的整体健康状况选择「很好」、「好」、「一般」、「不好」或「很不好」</li>
                    <li><strong>童年健康</strong>：回忆您童年时期的健康状况，选择「极好」、「很好」、「好」、「一般」或「不好」</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🏃 功能状态", expanded=False):
            st.markdown("""
            <div style="background-color: #fff8e1; border-radius: 10px; padding: 15px;">
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>ADL总分</strong>：日常生活活动能力总分（0-90分），点击「速测」按钮可辅助评估</li>
                    <li><strong>IADL总分</strong>：工具性日常生活活动能力总分（0-6分），点击「速测」按钮可辅助评估</li>
                </ul>
                <div style="margin-top: 10px; padding: 10px; background-color: #fff; border-radius: 8px; border-left: 4px solid #ff9800;">
                    <p style="margin: 0; font-size: 14px; color: #666;"><strong>提示</strong>：ADL和IADL速测可以帮助您更准确地评估功能状态</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🌙 生活方式", expanded=False):
            st.markdown("""
            <div style="background-color: #f3e5f5; border-radius: 10px; padding: 15px;">
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>夜间睡眠时长</strong>：使用滑块选择您的平均夜间睡眠时长（0-12小时）</li>
                    <li><strong>疼痛部位</strong>：选择您当前存在疼痛的部位，可多选，系统会自动计数</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🏥 慢性病管理", expanded=False):
            st.markdown("""
            <div style="background-color: #e8f5e9; border-radius: 10px; padding: 15px;">
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>慢性病</strong>：选择您目前患有的慢性病，可多选，系统会自动计数</li>
                    <li><strong>共病簇</strong>：系统会自动检测「胃病-关节炎」和「关节炎-哮喘」共病情况</li>
                </ul>
                <div style="margin-top: 10px; padding: 10px; background-color: #fff; border-radius: 8px; border-left: 4px solid #4caf50;">
                    <p style="margin: 0; font-size: 14px; color: #666;"><strong>提示</strong>：共病簇是指两种或多种慢性病同时存在的情况，可能对健康产生更大影响</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 评估结果说明
    st.markdown("## 📊 评估结果说明")
    result_container = st.container()
    with result_container:
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #e0e0e0;">
            <h3 style="margin: 0 0 15px 0; color: #333;">评估结果包含以下内容：</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                <div style="background-color: #e3f2fd; border-radius: 8px; padding: 15px;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <i class="fas fa-chart-pie" style="font-size: 20px; color: #1976d2; margin-right: 10px;"></i>
                        <h4 style="margin: 0; color: #1976d2;">风险概率</h4>
                    </div>
                    <p style="margin: 0; color: #666; font-size: 14px;">显示您的抑郁风险概率（0~1），数值越高风险越大</p>
                </div>
                <div style="background-color: #e8f5e9; border-radius: 8px; padding: 15px;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <i class="fas fa-tag" style="font-size: 20px; color: #388e3c; margin-right: 10px;"></i>
                        <h4 style="margin: 0; color: #388e3c;">风险类别</h4>
                    </div>
                    <p style="margin: 0; color: #666; font-size: 14px;">根据概率值判断：≥0.492 为「有风险」，否则为「无风险」</p>
                </div>
                <div style="background-color: #fff3e0; border-radius: 8px; padding: 15px;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <i class="fas fa-chart-bar" style="font-size: 20px; color: #f57c00; margin-right: 10px;"></i>
                        <h4 style="margin: 0; color: #f57c00;">影响因素分析</h4>
                    </div>
                    <p style="margin: 0; color: #666; font-size: 14px;">通过SHAP瀑布图展示各因素对风险评估的影响程度</p>
                </div>
                <div style="background-color: #f3e5f5; border-radius: 8px; padding: 15px;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <i class="fas fa-lightbulb" style="font-size: 20px; color: #7b1fa2; margin-right: 10px;"></i>
                        <h4 style="margin: 0; color: #7b1fa2;">个性化建议</h4>
                    </div>
                    <p style="margin: 0; color: #666; font-size: 14px;">根据评估结果提供针对性的健康建议</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 重要说明和技术信息
    st.markdown("## ⚠️ 重要说明")
    info_container = st.container()
    with info_container:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("""
            <div style="background-color: #fff3e0; border-radius: 12px; padding: 20px; height: 220px; border-left: 5px solid #ff9800;">
                <h3 style="margin: 0 0 5px 0; color: #e65100; font-size: 24px;">📋 医疗声明</h3>
                <ul style="margin: 0; padding-left: 20px; color: #666; line-height: 1.6;">
                    <li>本系统仅为<strong>抑郁风险初步筛查工具</strong></li>
                    <li>评估结果<strong>不替代专业医疗诊断</strong></li>
                    <li>如有身体不适或情绪困扰，请及时前往社区卫生服务中心或心理科进一步检查</li>
                    <li>本系统不存储个人身份信息，所有数据均为匿名处理</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background-color: #e8f5e9; border-radius: 12px; padding: 20px; height: 220px; border-left: 5px solid #4caf50;">
                <h3 style="margin: 0 0 5px 0; color: #2e7d32; font-size: 24px;">🛠️ 技术信息</h3>
                <ul style="margin: 0; padding-left: 20px; color: #666; line-height: 1.6;">
                    <li><strong>模型</strong>：CatBoost 分类器</li>
                    <li><strong>AUC</strong>：0.7805</li>
                    <li><strong>阈值</strong>：0.492 (Youden 指数)</li>
                    <li><strong>特征数量</strong>：13 个健康指标</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # 常见问题
    st.markdown("## ❓ 常见问题")
    faq_container = st.container()
    with faq_container:
        faqs = [
            {
                "question": "评估需要多长时间？",
                "answer": "填写完整信息通常需要2-3分钟，评估过程仅需几秒钟。"
            },
            {
                "question": "数据会被保存吗？",
                "answer": "系统默认不会保存个人身份信息，评估结果可选择保存到本地数据库，方便后续查看。"
            },
            {
                "question": "如何理解SHAP分析结果？",
                "answer": "SHAP值表示各因素对预测结果的贡献程度，正值表示增加风险，负值表示降低风险，绝对值越大影响越大。"
            },
            {
                "question": "评估结果准确吗？",
                "answer": "本系统基于科学研究数据开发，具有较好的预测能力，但最终诊断仍需专业医生确认。"
            }
        ]
        for faq in faqs:
            with st.expander(faq["question"]):
                st.markdown(f"<p style='color: #666; line-height: 1.6;'>{faq['answer']}</p>", unsafe_allow_html=True)

    # 快速体验提示
    st.markdown("""
    <div style="background-color: #e3f2fd; border-radius: 12px; padding: 20px; margin-top: 30px; display: flex; align-items: center;">
        <div style="background-color: #1976d2; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
            <i class="fas fa-lightbulb" style="font-size: 20px;"></i>
        </div>
        <div style="display: flex; align-items: center; height: 100%;">
            <p style="margin: 0; color: #666; font-size: 18px;">侧边栏点击「高风险示例」或「低风险示例」，系统将自动填充数据并开始评估。</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 页脚信息
    st.markdown("""
    <div style="background-color: #f8f9fa; border-radius: 12px; padding: 20px; margin-top: 30px; text-align: center;">
        <h3 style="margin: 0 0 15px 0; color: #333;">📞 技术支持</h3>
        <p style="margin: 0 0 10px 0; color: #666;">📧 support@example.com | 📞 400-xxx-xxxx</p>
        <hr style="margin: 15px 0;">
        <p style="margin: 0; font-size: 14px; color: #888;">版本 1.0.0 | © 2026 抑郁风险预警系统</p>
    </div>
    """, unsafe_allow_html=True)


def load_example_data(high_risk: bool = True):
    """加载示例数据"""
    if high_risk:
        # 高风险示例
        example_data = {
            'age': 72,
            'gender': '女',
            'education_level': '小学',
            'residence_type': '农村',
            'self_rated_health': '很不好',
            'childhood_health': '不好',
            'pain_sites': ['头', '背', '腰', '膝'],
            'sleep_hours_night': 4.5,
            'ADL_total': 45,
            'IADL_total': 2,
            'chronic_diseases': ['高血压', '糖尿病', '关节炎或风湿病'],
            'stomach_arthritis_pair': False,
            'arthritis_asthma_pair': False
        }
    else:
        # 低风险示例
        example_data = {
            'age': 65,
            'gender': '男',
            'education_level': '高中/中专/技校',
            'residence_type': '城镇',
            'self_rated_health': '很好',
            'childhood_health': '很好',
            'pain_sites': [],
            'sleep_hours_night': 7.5,
            'ADL_total': 90,
            'IADL_total': 6,
            'chronic_diseases': [],
            'stomach_arthritis_pair': False,
            'arthritis_asthma_pair': False
        }

    # 更新表单数据
    for key, value in example_data.items():
        st.session_state.form_data[key] = value

    st.success(f"已加载{'高风险' if high_risk else '低风险'}示例数据")


def reset_form():
    """重置表单"""
    st.session_state.form_data = DEFAULT_VALUES.copy()
    if 'last_prediction' in st.session_state:
        del st.session_state.last_prediction
    if 'show_result' in st.session_state:
        st.session_state.show_result = False


def perform_prediction(form_data: Dict) -> Optional[Dict]:
    """执行预测"""
    try:
        # 加载模型
        if 'model_data' not in st.session_state:
            with st.spinner("正在加载模型..."):
                st.session_state.model_data = load_model()

        model_data = st.session_state.model_data
        model = model_data['model']
        explainer = model_data['explainer']
        threshold = model_data['threshold']
        feature_names = model_data['feature_names']

        # 数据预处理
        with st.spinner("正在处理数据..."):
            # 使用 data_preprocess.preprocess_input 处理数据
            X = preprocess_input(form_data)
            
            # 获取特征值字典用于结果展示
            from utils.data_preprocess import encode_categorical_features
            features = encode_categorical_features(form_data)

        # 进行预测
        with st.spinner("正在进行风险评估..."):
            probability = model.predict_proba(X)[0, 1]  # 获取正类概率
            is_risk = probability >= threshold

        # 生成SHAP解释
        with st.spinner("正在生成解释..."):
            shap_values, base_value = generate_shap_values(explainer, X)

        # 准备结果
        result = {
            'probability': float(probability),
            'is_risk': bool(is_risk),
            'threshold': float(threshold),
            'feature_values': features,
            'shap_values': shap_values,
            'base_value': float(base_value),
            'feature_names': feature_names,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return result

    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")
        return None


def generate_pdf_report(record) -> bytes:
    """生成PDF报告"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from io import BytesIO
        import os
        import platform
        import json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # 自动检测系统字体
        def get_chinese_font():
            system = platform.system()
            if system == 'Windows':
                # Windows 系统字体路径
                font_paths = [
                    'C:\\Windows\\Fonts\\simhei.ttf',  # 黑体
                    'C:\\Windows\\Fonts\\simsun.ttc',  # 宋体
                    'C:\\Windows\\Fonts\\simkai.ttf',  # 楷体
                ]
            elif system == 'Darwin':  # macOS
                font_paths = [
                    '/Library/Fonts/STHeiti Light.ttc',  # 黑体
                    '/Library/Fonts/Songti.ttc',  # 宋体
                ]
            else:  # Linux
                font_paths = [
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # 文泉驿微米黑
                    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # 文泉驿正黑
                ]
            
            # 查找可用字体
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return font_path
            return None

        # 注册中文字体
        font_path = get_chinese_font()
        if font_path:
            font_name = 'ChineseFont'
            pdfmetrics.registerFont(TTFont(font_name, font_path))
        else:
            # 如果没有找到中文字体，使用默认字体
            font_name = 'Helvetica'

        # 创建PDF文档
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []

        # 样式
        styles = getSampleStyleSheet()
        
        # 创建自定义样式，指定中文字体
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=font_name,
            fontSize=24,
            spaceAfter=20
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=18,
            spaceAfter=12
        )
        
        normal_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontName=font_name,
            fontSize=12,
            leading=18
        )

        # 添加标题
        elements.append(Paragraph("抑郁风险评估报告", title_style))
        elements.append(Spacer(1, 1*cm))

        # 添加基本信息
        elements.append(Paragraph("1. 基本信息", heading_style))
        elements.append(Spacer(1, 0.5*cm))
        
        # 表格数据，使用Paragraph确保中文显示
        data = [
            [Paragraph('项目', normal_style), Paragraph('值', normal_style)],
            [Paragraph('时间', normal_style), Paragraph(str(record['timestamp']), normal_style)],
            [Paragraph('备注', normal_style), Paragraph(record['nickname'] if record['nickname'] else '无', normal_style)],
            [Paragraph('风险概率', normal_style), Paragraph(f"{record['probability']:.3f}", normal_style)],
            [Paragraph('风险类别', normal_style), Paragraph(record['risk_level'], normal_style)]
        ]
        
        table = Table(data, colWidths=[6*cm, 10*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 1*cm))

        # 添加健康信息输入
        elements.append(Paragraph("2. 健康信息输入", heading_style))
        elements.append(Spacer(1, 0.5*cm))
        
        # 解析输入特征
        try:
            input_features_raw = record.get('input_features', None)
            
            # 情况1：None或空值
            if input_features_raw is None or input_features_raw == '' or (isinstance(input_features_raw, str) and input_features_raw.strip() == ''):
                input_features = {}
            # 情况2：已经是字典（某些情况下pandas可能自动解析）
            elif isinstance(input_features_raw, dict):
                input_features = input_features_raw
            # 情况3：字节串（需要解码）
            elif isinstance(input_features_raw, bytes):
                try:
                    input_features = json.loads(input_features_raw.decode('utf-8'))
                except:
                    input_features = {}
            # 情况4：字符串（正常情况，需要JSON解析）
            elif isinstance(input_features_raw, str):
                try:
                    # 尝试JSON解析
                    # 先去除可能的前后引号
                    clean_str = input_features_raw.strip()
                    if clean_str.startswith('"') and clean_str.endswith('"'):
                        clean_str = clean_str[1:-1]
                    
                    # 第一次尝试解析
                    input_features = json.loads(clean_str)
                    
                    # 检查是否是双重序列化（解析后仍然是字符串）
                    if isinstance(input_features, str):
                        try:
                            # 第二次尝试解析
                            input_features = json.loads(input_features)
                        except:
                            # 解析失败，使用空字典
                            input_features = {}
                            
                except Exception as e:
                    # 解析失败，尝试更宽松的处理
                    try:
                        # 尝试处理可能的转义问题
                        import ast
                        input_features = ast.literal_eval(clean_str)
                        if not isinstance(input_features, dict):
                            input_features = {}
                    except:
                        # 所有尝试都失败，使用空字典
                        input_features = {}
            # 情况5：其他类型，使用空字典
            else:
                input_features = {}
                
        except Exception as e:
            # 任何异常都使用空字典
            input_features = {}
        
        # 特征中文名称映射
        feature_names_cn = {
            'gender': '性别',
            'age': '年龄',
            'edu_years': '受教育年限',
            'residence_type': '居住地类型',
            'self_rated_health': '自评健康',
            'childhood_health': '童年健康',
            'pain_site_count': '疼痛部位数量',
            'ADL_total': 'ADL总分',
            'IADL_total': 'IADL总分',
            'chronic_count': '慢性病数量',
            'stomach_arthritis_pair': '胃病-关节炎共病',
            'arthritis_asthma_pair': '关节炎-哮喘共病',
            'sleep_hours_night': '夜间睡眠时长'
        }
        
        # 准备特征表格数据
        feature_data = [[Paragraph('特征', normal_style), Paragraph('值', normal_style)]]
        
        # 确保input_features是字典
        if isinstance(input_features, dict):
            if input_features:
                # 按预设顺序显示特征
                feature_order = [
                    'age', 'gender', 'edu_years', 'residence_type',
                    'self_rated_health', 'childhood_health', 'pain_site_count',
                    'ADL_total', 'IADL_total', 'chronic_count',
                    'stomach_arthritis_pair', 'arthritis_asthma_pair', 'sleep_hours_night'
                ]
                
                for feature in feature_order:
                    if feature in input_features:
                        value = input_features[feature]
                        feature_cn = feature_names_cn.get(feature, feature)
                        # 处理特殊值
                        if feature == 'gender':
                            value_str = '女' if value == 1 else '男'
                        elif feature == 'residence_type':
                            value_str = '城镇' if value == 1 else '农村'
                        elif feature in ['stomach_arthritis_pair', 'arthritis_asthma_pair']:
                            value_str = '是' if value == 1 else '否'
                        else:
                            value_str = str(value)
                        feature_data.append([Paragraph(feature_cn, normal_style), Paragraph(value_str, normal_style)])
            else:
                # 如果是字典但为空，显示提示信息
                feature_data.append([Paragraph('无输入特征数据', normal_style), Paragraph('', normal_style)])
        else:
            # 如果不是字典，显示空数据
            feature_data.append([Paragraph('无输入特征数据', normal_style), Paragraph('', normal_style)])
        
        # 创建特征表格
        feature_table = Table(feature_data, colWidths=[6*cm, 10*cm])
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(feature_table)
        elements.append(Spacer(1, 1*cm))

        # 添加SHAP瀑布图
        elements.append(Paragraph("3. 风险因素分析", heading_style))
        elements.append(Spacer(1, 0.5*cm))
        
        try:
            # 加载模型和计算SHAP值
            from utils.shap_utils import create_shap_waterfall_plot, get_feature_names_cn
            from utils.data_preprocess import preprocess_input
            
            # 重建特征数据
            feature_values = input_features
            
            # 从配置文件获取模型路径
            model_path = CONFIG.get('model', {}).get('model_path', 'models/CatBoost_F2_v1_with_threshold.pkl')
            # 使用绝对路径加载模型文件
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
            loaded_model = joblib.load(model_path)
            model = loaded_model['model']
            
            # 从Pipeline中提取真正的CatBoost模型
            from utils.shap_utils import extract_catboost_model
            model = extract_catboost_model(model)
            
            # 准备特征顺序
            feature_order = [
                'self_rated_health', 'pain_site_count', 'sleep_hours_night',
                'IADL_total', 'ADL_total', 'edu_years', 'childhood_health',
                'residence_type', 'gender', 'chronic_count',
                'stomach_arthritis_pair', 'arthritis_asthma_pair', 'age'
            ]
            
            # 构建特征向量
            X = []
            for feat in feature_order:
                if isinstance(feature_values, dict):
                    X.append(feature_values.get(feat, 0))
                else:
                    X.append(0)
            X = np.array(X).reshape(1, -1)
            
            # 计算SHAP值
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values, _ = generate_shap_values(explainer, X)
            
            # 生成SHAP图
            fig = create_shap_waterfall_plot(
                shap_values[0],
                feature_order,
                explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            )
            
            if fig:
                # 将图表保存到BytesIO
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                
                # 添加图片到PDF
                elements.append(Image(img_buffer, width=16*cm, height=10*cm))
                plt.close()
            else:
                elements.append(Paragraph("无法生成风险因素分析图", normal_style))
        except Exception as e:
            elements.append(Paragraph(f"生成风险因素分析图时出错: {str(e)}", normal_style))

        elements.append(Spacer(1, 1*cm))

        # 添加个性化建议
        elements.append(Paragraph("4. 个性化健康建议", heading_style))
        elements.append(Spacer(1, 0.5*cm))
        
        try:
            from utils.shap_utils import generate_text_explanation
            
            # 生成个性化建议
            risk_info = {
                'is_risk': record['risk_level'] == '有风险',
                'risk_category': record['risk_level']
            }
            
            # 生成文字解释
            if 'shap_values' in locals():
                try:
                    explanation = generate_text_explanation(
                        shap_values[0],
                        feature_order,
                        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                        record['probability']
                    )
                    elements.append(Paragraph(explanation, normal_style))
                except Exception as e:
                    elements.append(Paragraph(f"生成文字解释时出错: {str(e)}", normal_style))
                    # 回退到简单建议
                    if record['risk_level'] == '有风险':
                        elements.append(Paragraph("建议您关注自身心理健康，保持规律作息，适当运动，多与家人朋友交流。如症状持续，建议咨询专业心理医生。", normal_style))
                    else:
                        elements.append(Paragraph("您的抑郁风险较低，请继续保持健康的生活方式，定期进行健康检查。", normal_style))
            else:
                # 简单的建议
                if record['risk_level'] == '有风险':
                    elements.append(Paragraph("建议您关注自身心理健康，保持规律作息，适当运动，多与家人朋友交流。如症状持续，建议咨询专业心理医生。", normal_style))
                else:
                    elements.append(Paragraph("您的抑郁风险较低，请继续保持健康的生活方式，定期进行健康检查。", normal_style))
        except Exception as e:
            elements.append(Paragraph(f"生成个性化建议时出错: {str(e)}", normal_style))

        elements.append(Spacer(1, 1*cm))

        # 添加评估说明
        elements.append(Paragraph("5. 评估说明", heading_style))
        elements.append(Spacer(1, 0.5*cm))
        elements.append(Paragraph("本报告基于CatBoost机器学习模型对抑郁风险进行评估。评估结果仅供参考，不替代专业医疗诊断。", normal_style))
        elements.append(Spacer(1, 0.5*cm))
        elements.append(Paragraph("如评估结果显示高风险，建议及时咨询专业医生或心理卫生专家。", normal_style))

        # 添加免责声明
        elements.append(Spacer(1, 1*cm))
        elements.append(Paragraph("6. 免责声明", heading_style))
        elements.append(Spacer(1, 0.5*cm))
        elements.append(Paragraph("本系统仅为抑郁风险初步筛查工具，不替代专业医疗诊断。评估结果仅供参考，如有身体不适或情绪困扰，请及时前往社区卫生服务中心或心理科进一步检查。", normal_style))

        # 生成PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
    except ImportError as e:
        st.error(f"生成PDF报告需要安装reportlab库，请运行: pip install reportlab。错误: {str(e)}")
        return None
    except Exception as e:
        import traceback
        st.error(f"生成PDF报告时发生错误: {str(e)}")
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None


def batch_export_pdf(records: pd.DataFrame) -> bytes:
    """
    批量生成PDF报告并打包为ZIP文件

    Args:
        records: 包含预测记录的DataFrame

    Returns:
        ZIP文件的字节数据
    """
    try:
        from io import BytesIO
        import datetime

        # 创建ZIP文件的内存缓冲区
        zip_buffer = BytesIO()
        
        # 创建ZIP文件
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 为每条记录生成PDF
            for _, row in records.iterrows():
                # 生成PDF报告
                pdf_data = generate_pdf_report(row.to_dict())
                if pdf_data:
                    # 生成文件名
                    timestamp_str = str(row['timestamp'])
                    safe_timestamp = timestamp_str.replace(':', '-').replace(' ', '_')
                    filename = f"depression_report_{safe_timestamp}.pdf"
                    # 添加到ZIP文件
                    zf.writestr(filename, pdf_data)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    except Exception as e:
        import traceback
        st.error(f"批量生成PDF时发生错误: {str(e)}")
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None


def save_prediction_to_db(result: Dict, nickname: str = "") -> bool:
    """保存预测结果到数据库"""
    try:
        # 准备数据
        data = {
            'nickname': nickname,
            'probability': result['probability'],
            'risk_category': '有风险' if result['is_risk'] else '无风险',
            'input_features': result['feature_values'],
            'timestamp': datetime.now()
        }

        # 保存到数据库
        success = save_prediction(data)
        return success

    except Exception as e:
        st.error(f"保存失败: {str(e)}")
        return False


# ========== 主程序 ==========
def main():
    """主函数"""
    # 初始化session state
    if 'form_data' not in st.session_state:
        st.session_state.form_data = DEFAULT_VALUES.copy()

    if 'page' not in st.session_state:
        st.session_state.page = "🔍 风险评估"

    if 'show_result' not in st.session_state:
        st.session_state.show_result = False

    if 'selected_ids' not in st.session_state:
        st.session_state.selected_ids = []

    # 自定义CSS
    st.markdown("""
    <style>
    /* 大字体，适合中老年用户 */
    .stNumberInput input, .stTextInput input, .stSelectbox select, .stRadio label, .stCheckbox label, .stSlider label {
        font-size: 20px !important;
    }

    /* 按钮样式 */
    .stButton > button {
        font-size: 18px;
        padding: 10px 20px;
    }
    
    /* 标签文字大小 */
    .stMarkdown h3, .stMarkdown p {
        font-size: 18px !important;
    }

    /* 风险等级样式 */
    .risk-high {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }

    .risk-medium {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }

    .risk-low {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* 侧边栏样式 */
    .css-1d391kg {
        padding-top: 2rem;
    }

    /* 响应式调整 */
    @media (max-width: 768px) {
        .stColumn {
            width: 100% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # 渲染侧边栏
    page = render_sidebar()

    # 根据选择的页面渲染相应内容
    if page == "🔍 风险评估":
        render_assessment_page()
    elif page == "📊 历史记录":
        render_history_page()
    elif page == "🧠 模型解释":
        render_global_explanation_page()
    else:  # "ℹ️ 使用说明"
        render_instructions_page()

    # 自动评估（用于示例数据）
    if st.session_state.get('auto_evaluate', False):
        st.session_state.auto_evaluate = False
        # 模拟表单提交
        with st.spinner("正在评估示例数据..."):
            result = perform_prediction(st.session_state.form_data)
            if result:
                st.session_state.last_prediction = result
                st.session_state.show_result = True
                st.rerun()


if __name__ == "__main__":
    # 初始化数据库
    init_db()

    # 运行主程序
    main()
