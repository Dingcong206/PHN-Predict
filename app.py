import joblib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for

# 1. 模型加载 (Model Loading)
MODEL_PATH = 'final_model.joblib'
PREPROCESS_FILE = 'preprocess_pipeline.joblib'

try:
    # 应用程序启动时加载模型，只需加载一次
    model = joblib.load(MODEL_PATH)
    preprocess = joblib.load(PREPROCESS_FILE)
    print("Model loaded successfully.")
except Exception as e:
    # 如果模型加载失败，打印错误并退出
    print(f"Model loading failed: {e}")
    model = None

app = Flask(__name__)


# 定义一个帮助函数来处理空字符串、None 和转换为 float，同时确保 Yes/No 转换为 1/0
def get_feature_value(field_name, default_value=0.0):
    """
    从表单中获取字段值，处理空值，并将分类变量转换为浮点数。
    """
    value = request.form.get(field_name)
    if value is None or value.strip() == '' or value.strip() == 'Select Gender' or value.strip() == 'Select Status (Yes/No)':
        return default_value

    # 分类变量转换为 1/0
    if value in ('Yes', '1'):
        return 1.0
    # 女性是 '2'，
    if value in ('No', '2'):
        return 0.0

    # 数值变量转换为 float
    try:
        return float(value)
    except ValueError:
        print(f"Warning: Could not convert input for {field_name} to float: {value}")
        return default_value


# --- 路由 1: 输入表单页面  ---
@app.route('/')
def home():
    """渲染指标输入表单页面"""
    return render_template('index.html')


# --- 路由 2: 预测处理 (Route for Prediction) ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "模型未加载，无法预测。", 500


    try:
        # 1. 仅获取模型所需的 7 个特征，并严格按照训练时的顺序排列：
        # ["Age", "Baseline_VAS", "PCS", "PSQI", "MCS", "RBC", "APTT(time)"]

        input_features = [
            get_feature_value('age'),  # Age
            get_feature_value('pcs_score'),  # PCS
            get_feature_value('mcs_ics_score'),  # MCS
            get_feature_value('psqi_score'),  # PSQI
            get_feature_value('rbc'),  # RBC
            get_feature_value('aptt_time'),  # APTT(time)
            get_feature_value('baseline_vas_score'),  # Baseline_VAS
        ]

        # 2. 数据预处理
        # 转换为模型需要的 2D numpy 数组格式: [[feature1, feature2, ...]]
        final_features = np.array(input_features).reshape(1, -1)

        scaler = preprocess.named_transformers_["num"]
        means = scaler.mean_[[0, 4, 5, 6, 18, 33]]
        scales = scaler.scale_[[0, 4, 5, 6, 18, 33]]

        x_num = test_data[:, :6]
        x_cat = test_data[:, [-1]]

        x_num_scaled = (x_num - means) / scales
        x_final = np.concatenate([x_num_scaled, x_cat], axis=1)

        # 3. 进行预测
        prediction_proba = model.predict_proba(x_final)[:, 1]  # 取 PHN 阳性的概率 (第二列)
        prediction_class = model.predict(x_final)[0]  # 取预测的类别 (0 或 1)

        # 4. 跳转到结果页面，传递类别和概率

        return redirect(url_for('result',
                                prediction_class=str(prediction_class),
                                prediction_proba=f"{prediction_proba[0]:.4f}"))

    except Exception as e:
        # 如果获取数据或转换失败，返回错误信息
        print(f"Prediction logic failed: {e}")
        return f"Prediction failed. Please check the input format or feature processing logic.：{e}", 400


# --- 路由 3: 结果展示页面 ---
@app.route('/result')
def result():
    """显示预测结果页面"""
    # 从 URL 参数中获取预测结果
    prediction_class = request.args.get('prediction_class', 'N/A')
    prediction_proba = request.args.get('prediction_proba', 'N/A')

    # 将类别结果转换为易读的文本
    if prediction_class == '1':
        result_text = "high Risk"
    elif prediction_class == '0':
        result_text = "Low Risk"
    else:
        result_text = "Abnormal prediction results"

    # 渲染结果页面
    return render_template('results.html',
                           prediction_class_text=result_text,
                           prediction_proba=prediction_proba)


# 启动应用
if __name__ == '__main__':

    app.run(debug=True)