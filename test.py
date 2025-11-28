import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    auc as skl_auc,  # 导入 sklearn.metrics.auc
    precision_recall_curve  # 用于 PR-AUC
)
from sklearn.exceptions import InconsistentVersionWarning

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- 1. 配置 ---
DATA_FILE = '外部数据集11.2.xlsx'
MODEL_FILE = 'final_model.joblib'
PREPROCESS_FILE = 'preprocess_pipeline.joblib'
TARGET_COLUMN = 'PHN'  # 目标列名

# 7 个最终输入到模型的特征，请确保顺序正确！
KEY_FEATURES = ["Age", "PCS", "MCS", "PSQI", "RBC", "APTT(time)", "Baseline_VAS"]
NUMERICAL_FEATURES = KEY_FEATURES[:6]  # 假设前 6 个是数值型
CATEGORICAL_FEATURE = KEY_FEATURES[-1]  # 假设最后一个是类别型


# =========================================================================
# === 修正函数：解决 joblib 文件的编码问题 ===
# =========================================================================
def safe_joblib_load(file_path):
    try:
        return joblib.load(file_path)
    except UnicodeDecodeError as e:
        if 'utf-8' in str(e) and 'invalid start byte' in str(e):
            print(f"    ⚠️ 警告：加载 {file_path} 时检测到 UTF-8 编码错误，尝试使用 'latin-1' 编码。")
            return joblib.load(file_path, mmap_mode=None, encoding='latin1')
        raise e
    except Exception as e:
        if "No such file or directory" in str(e):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        raise e


# =========================================================================

print("--- 模型准确率测试开始 ---")

# --- 2. 加载模型和数据 ---
try:
    final_model = safe_joblib_load(MODEL_FILE)
    preprocess = safe_joblib_load(PREPROCESS_FILE)
    data = pd.read_excel(DATA_FILE)
    data.columns = data.columns.str.strip()  # 清理列名空格

    print("✅ 模型和预处理器加载成功。")
    print(f"✅ 数据集加载成功，共 {len(data)} 条记录。")

except Exception as e:
    print(f"❌ 错误：加载文件时发生异常: {e}")
    exit(1)

# --- 3. 手动执行数据预处理和预测 (基于您的逻辑) ---
try:
    # 3.1 提取标签和关键特征
    y_true = data[TARGET_COLUMN]
    X_raw = data[KEY_FEATURES].values  # 提取并转换为 numpy 数组

    # 从 ColumnTransformer 中提取 Scaler 对象
    scaler = preprocess.named_transformers_["num"]

    # ⚠️ 关键：这里使用的索引 [0, 4, 5, 6, 18, 33] 必须与模型训练脚本一致！
    means = scaler.mean_[[0, 4, 5, 6, 18, 33]]
    scales = scaler.scale_[[0, 4, 5, 6, 18, 33]]

    # 分割数值和类别特征
    x_num = X_raw[:, :len(NUMERICAL_FEATURES)]
    x_cat = X_raw[:, [-1]]

    # 缩放数值特征
    x_num_scaled = (x_num.astype(float) - means) / scales

    # 合并为最终的 7 个特征输入
    x_final = np.concatenate([x_num_scaled, x_cat], axis=1)

    # 预测
    y_prob = final_model.predict_proba(x_final)[:, 1]
    y_pred = final_model.predict(x_final)

except Exception as e:
    print(f"\n❌ 预测或手动预处理失败。")
    print(f"原始错误: {e}")
    exit(1)

# --- 4. 结果计算和输出 (满足您的格式要求) ---
print("\n--- 结果计算 ---")

# 混淆矩阵 (CM)
cm = confusion_matrix(y_true, y_pred)
# TN: True Negative, FP: False Positive, FN: False Negative, TP: True Positive
TN, FP, FN, TP = cm.ravel()

# 指标计算
accuracy = accuracy_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred)  # 灵敏度 = 召回率 (TPR)
specificity = TN / (TN + FP) if (TN + FP) else 0  # 特异度 = TN / (TN + FP)
ppv = precision_score(y_true, y_pred)  # PPV = 精确率
npv = TN / (TN + FN) if (TN + FN) else 0  # NPV = TN / (TN + FN)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

# 计算 PR-AUC (Precision-Recall AUC)
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
pr_auc = skl_auc(recall_vals, precision_vals)

# 格式化输出
print("\n[External Validation - Final Model Only]")
print(f"Model:       ExtraTrees")
print(f"Accuracy:    {accuracy:.3f}")
print(f"AUC:         {auc:.3f}")
print(f"PR-AUC:      {pr_auc:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"PPV:         {ppv:.3f}")
print(f"NPV:         {npv:.3f}")
print(f"F1:          {f1:.3f}")

print("\nConfusion Matrix:")
# 打印混淆矩阵，格式尽量与您的要求对齐
print(f"[[{TN:^3}  {FP:^3}]")
print(f" [ {FN:^3}  {TP:^3}]]")

print("\n--- 测试完成 ---")