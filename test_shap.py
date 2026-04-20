#!/usr/bin/env python3
"""
测试SHAP解释功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("开始测试SHAP解释功能...")
print("当前工作目录:", os.getcwd())
print("Python版本:", sys.version)

# 测试导入
print("\n=== 测试导入 ===")
try:
    import numpy as np
    print("NumPy 导入成功")
except Exception as e:
    print("NumPy 导入失败:", e)

try:
    import pandas as pd
    print("Pandas 导入成功")
except Exception as e:
    print("Pandas 导入失败:", e)

try:
    import shap
    print("SHAP 导入成功")
except Exception as e:
    print("SHAP 导入失败:", e)

try:
    import matplotlib.pyplot as plt
    print("Matplotlib 导入成功")
except Exception as e:
    print("Matplotlib 导入失败:", e)

try:
    from utils.shap_utils import (
        generate_shap_values, create_shap_waterfall_plot,
        generate_text_explanation, get_feature_names_cn,
        create_interactive_shap_waterfall, generate_feature_importance_bar,
        generate_dependence_plot
    )
    print("SHAP工具函数导入成功")
except Exception as e:
    print("SHAP工具函数导入失败:", e)
    import traceback
    traceback.print_exc()

print("\n测试完成！")
