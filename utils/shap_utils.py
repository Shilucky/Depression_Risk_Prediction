# utils/shap_utils.py
"""
SHAP 可解释性分析模块
功能：生成 SHAP 值、瀑布图、文字解释、依赖图等
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import joblib

# 特征中文名称映射（与 data_preprocess 保持一致）
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


def get_feature_names_cn() -> Dict[str, str]:
    """返回特征中文名称映射"""
    return FEATURE_NAMES_CN.copy()


def extract_catboost_model(pipeline_or_model):
    """
    从 Pipeline 中提取真正的 CatBoost 分类器（用于 SHAP）
    如果输入已经是模型，则直接返回
    """
    if hasattr(pipeline_or_model, 'named_steps') and 'clf' in pipeline_or_model.named_steps:
        return pipeline_or_model.named_steps['clf']
    return pipeline_or_model


def load_model_for_shap(model_path: str):
    """
    加载模型文件（支持打包的字典或直接模型），并返回 CatBoost 分类器
    """
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and 'model' in loaded:
        pipeline = loaded['model']
    else:
        pipeline = loaded
    return extract_catboost_model(pipeline)


def generate_shap_values(explainer, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    计算 SHAP 值并返回处理后的正类 SHAP 值和基准值
    增强版：兼容 shap.Explanation 对象、列表、三维数组等格式

    Args:
        explainer: shap.TreeExplainer 实例
        X: 输入特征 DataFrame (单样本或多样本)

    Returns:
        (shap_values, base_value)
        - shap_values: 处理后的二维数组 (n_samples, n_features)，对应正类（抑郁风险）的 SHAP 值
        - base_value: 基准值（正类的期望概率）
    """
    shap_values_raw = explainer.shap_values(X)
    expected_value = explainer.expected_value

    # 处理 shap.Explanation 对象
    if hasattr(shap_values_raw, 'values'):
        shap_values_raw = shap_values_raw.values

    # 处理多分类/二分类输出格式
    if isinstance(shap_values_raw, list):
        # 二分类：列表 [负类SHAP, 正类SHAP]
        shap_values = shap_values_raw[1]
    elif len(shap_values_raw.shape) == 3:
        # 三维数组 (n_samples, n_features, n_classes)，取正类（索引1）
        shap_values = shap_values_raw[:, :, 1]
    else:
        shap_values = shap_values_raw

    # 处理基准值（可能是数组，取正类）
    if isinstance(expected_value, (list, np.ndarray)):
        base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    else:
        base_value = expected_value

    return shap_values, base_value


def create_shap_waterfall_plot(
    shap_values: np.ndarray,
    feature_names: List[str],
    base_value: float,
    max_display: int = 13,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """
    生成单个样本的 SHAP 瀑布图（matplotlib 图形）

    Args:
        shap_values: 单个样本的 SHAP 值，形状 (n_features,)
        feature_names: 特征名称列表（英文）
        base_value: 基准值（正类的期望概率）
        max_display: 最多显示的特征数量
        figsize: 图形大小

    Returns:
        matplotlib Figure 对象，失败时返回 None
    """
    if shap_values is None or len(shap_values) == 0:
        return None

    # 确保 shap_values 是一维数组
    if len(shap_values.shape) > 1:
        shap_values = shap_values.flatten()

    # 使用全局字体设置
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 将英文特征名称转换为中文
    name_map = get_feature_names_cn()
    chinese_feature_names = [name_map.get(feat, feat) for feat in feature_names]
    
    # 创建 Explanation 对象
    exp = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=np.zeros_like(shap_values),  # 原始数据值（仅用于显示，瀑布图中不展示具体数值）
        feature_names=chinese_feature_names
    )

    try:
        plt.figure(figsize=figsize)
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"生成瀑布图失败: {e}")
        return None


def generate_text_explanation(
    shap_values_single: np.ndarray,
    feature_names: List[str],
    base_value: float,
    probability: float,
    top_n: int = 3
) -> str:
    """
    根据 SHAP 值生成自然语言解释

    Args:
        shap_values_single: 单个样本的 SHAP 值 (n_features,)
        feature_names: 特征名称列表（英文）
        base_value: 基准值
        probability: 模型预测概率
        top_n: 分别展示正向和负向贡献最大的特征数

    Returns:
        解释文本（Markdown 格式）
    """
    if shap_values_single is None or len(shap_values_single) == 0:
        return "无法生成解释信息。"

    name_map = get_feature_names_cn()

    contributions = []
    for i, feat in enumerate(feature_names):
        if i < len(shap_values_single):
            contributions.append((feat, shap_values_single[i]))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    positive = [(feat, val) for feat, val in contributions if val > 0]
    negative = [(feat, val) for feat, val in contributions if val < 0]

    top_positive = positive[:top_n]
    top_negative = negative[:top_n]

    lines = []
    lines.append(f"**模型预测概率**: {probability:.3f}（基准值: {base_value:.3f}）\n")
    lines.append("### 主要风险因素（增加风险）")
    if top_positive:
        for feat, val in top_positive:
            cn_name = name_map.get(feat, feat)
            lines.append(f"- **{cn_name}**: 贡献 {val:+.3f}")
    else:
        lines.append("- 无明显增加风险的因素")

    lines.append("\n### 主要保护因素（降低风险）")
    if top_negative:
        for feat, val in top_negative:
            cn_name = name_map.get(feat, feat)
            lines.append(f"- **{cn_name}**: 贡献 {val:+.3f}")
    else:
        lines.append("- 无明显降低风险的因素")

    if probability >= 0.5:
        lines.append("\n**总体评估**：当前评估结果提示存在抑郁风险，建议关注上述高风险因素，并咨询专业医生。")
    else:
        lines.append("\n**总体评估**：当前评估结果提示抑郁风险较低，请继续保持健康生活方式。")

    return "\n".join(lines)


def generate_feature_importance_bar(
    shap_values: np.ndarray,
    feature_names: List[str],
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """
    生成全局特征重要性条形图（所有样本的平均 SHAP 绝对值）

    Args:
        shap_values: 多维 SHAP 值，形状 (n_samples, n_features)
        feature_names: 特征名称列表（英文）
        figsize: 图形大小

    Returns:
        matplotlib Figure 对象
    """
    if shap_values is None or len(shap_values) == 0:
        return None

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    indices = np.argsort(mean_abs_shap)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = mean_abs_shap[indices]

    name_map = get_feature_names_cn()
    sorted_names_cn = [name_map.get(n, n) for n in sorted_names]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names_cn)
    ax.invert_yaxis()
    ax.set_xlabel('平均 |SHAP 值|')
    ax.set_title('全局特征重要性')
    plt.tight_layout()
    return fig


def generate_dependence_plot(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature_name: str,
    feature_names_cn: Dict[str, str],
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    生成特征依赖图（SHAP Dependence Plot）

    Args:
        shap_values: SHAP 值数组，形状 (n_samples, n_features)
        X_sample: 样本特征 DataFrame（与 shap_values 对应）
        feature_name: 要绘制的特征名（英文）
        feature_names_cn: 特征中文名映射字典
        figsize: 图形大小
        save_path: 可选，保存路径

    Returns:
        matplotlib Figure 对象
    """
    if feature_name not in X_sample.columns:
        print(f"特征 {feature_name} 不存在于数据中")
        return None

    try:
        plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature_name, shap_values, X_sample,
            show=False, interaction_index='auto'
        )
        ax = plt.gca()
        cn_name = feature_names_cn.get(feature_name, feature_name)
        ax.set_xlabel(cn_name, fontsize=12)
        ax.set_ylabel('SHAP 值', fontsize=12)
        plt.title(f'特征依赖图: {cn_name}', fontsize=14)

        # 修改颜色条标签（如果存在交互特征）
        fig = plt.gcf()
        for cax in fig.axes:
            if cax != ax and cax.get_ylabel():
                interaction_en = cax.get_ylabel()
                interaction_cn = feature_names_cn.get(interaction_en, interaction_en)
                cax.set_ylabel(interaction_cn, fontsize=10)
                break

        plt.tight_layout()
        fig = plt.gcf()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    except Exception as e:
        print(f"生成依赖图失败: {e}")
        return None


def create_interactive_shap_waterfall(
    shap_values: np.ndarray,
    feature_names: List[str],
    base_value: float,
    probability: float,
    max_display: int = 13
) -> 'plotly.graph_objects.Figure':
    """
    创建交互式SHAP瀑布图（Plotly图表）

    Args:
        shap_values: 单个样本的SHAP值，形状 (n_features,)
        feature_names: 特征名称列表（英文）
        base_value: 基准值（正类的期望概率）
        probability: 模型预测概率
        max_display: 最多显示的特征数量

    Returns:
        Plotly Figure 对象
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    # 确保shap_values是一维数组
    if len(shap_values.shape) > 1:
        shap_values = shap_values.flatten()

    # 将英文特征名称转换为中文
    name_map = get_feature_names_cn()
    chinese_feature_names = [name_map.get(feat, feat) for feat in feature_names]

    # 计算累积贡献
    contributions = list(zip(chinese_feature_names, shap_values))
    # 按绝对值排序，取前max_display个
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_contributions = contributions[:max_display]

    # 准备数据
    labels = [f"{name}" for name, val in top_contributions]
    values = [val for name, val in top_contributions]

    # 计算累积值
    cumulative = [base_value]
    for val in values:
        cumulative.append(cumulative[-1] + val)

    # 为每个条形设置颜色
    colors = []
    for val in values:
        if val >= 0:
            colors.append('#FF6B6B')  # 红色表示增加风险
        else:
            colors.append('#4ECDC4')  # 绿色表示降低风险

    # 创建瀑布图数据
    fig = go.Figure(go.Waterfall(
        name="SHAP贡献",
        orientation="v",
        measure=["relative"] * len(values) + ["total"],
        x=labels + ["最终概率"],
        textposition="outside",
        text=[f"{val:+.3f}" for val in values] + [f"{probability:.3f}"],
        y=values + [probability - cumulative[-2]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        base=base_value,
    ))

    # 更新条形颜色
    for i, color in enumerate(colors):
        fig.update_traces(selector=dict(type='waterfall', x=labels[i]), marker_color=color)
    # 最终概率条的颜色
    fig.update_traces(selector=dict(type='waterfall', x="最终概率"), marker_color='#45B7D1')

    # 添加基准值参考线
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=base_value,
        x1=len(labels) + 0.5,
        y1=base_value,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )

    # 添加累积线
    fig.add_trace(go.Scatter(
        x=[-0.5] + list(range(len(labels))) + [len(labels)],
        y=cumulative + [probability],
        mode='lines+markers',
        name='累积贡献',
        line=dict(color='#9467BD', width=2),
        marker=dict(color='#9467BD', size=6)
    ))

    # 更新布局
    fig.update_layout(
        title="SHAP 影响因素分析",
        xaxis_title="特征",
        yaxis_title="贡献值",
        template="plotly_white",
        height=600,
        margin=dict(l=100, r=50, t=80, b=200),
        hovermode="x unified",
        xaxis_tickangle=-45,
        showlegend=True
    )

    # 添加悬停信息
    fig.update_traces(
        hovertemplate=
        "特征: %{x}<br>" +
        "贡献: %{y:.3f}<br>" +
        "累积: %{customdata:.3f}<br>" +
        "<extra></extra>",
        customdata=cumulative[1:]
    )

    return fig


# 测试函数（可选）
if __name__ == "__main__":
    print("SHAP 工具模块测试")
    print("特征中文映射:", get_feature_names_cn())