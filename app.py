from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


# 模拟模型预测结果和 SHAP 数据 (实际应用中，这里会调用你的PHN模型)
def mock_phn_prediction(data):
    # 模拟预测结果：风险等级和概率
    import random
    risk_level = random.choice(['High Risk', 'Low Risk'])
    probability = round(random.uniform(90.0, 99.99), 2) if risk_level == 'High Risk' else round(
        random.uniform(5.0, 30.0), 2)

    # 模拟SHAP值数据 (用于生成可解释性图表的数据)
    shap_data = [
        {'feature': 'Age', 'value': 1.51, 'effect': 'Positive'},
        {'feature': 'WBC', 'value': 0.85, 'effect': 'Positive'},
        {'feature': 'pH', 'value': -0.41, 'effect': 'Negative'},
        {'feature': 'HCT', 'value': 0.37, 'effect': 'Positive'},
        {'feature': 'GLU', 'value': -0.16, 'effect': 'Negative'},
        # ... 更多特征
    ]

    return {
        'risk_level': 'PHN ' + risk_level,
        'probability': probability,
        'color_class': 'risk-high' if risk_level == 'High Risk' else 'risk-low',
        'shap_data': shap_data
    }


@app.route('/', methods=['GET'])
def index():
    # 渲染输入表单页面
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 1. 获取表单数据
    form_data = request.form

    # 2. 运行模型模拟
    result = mock_phn_prediction(form_data)

    # 3. 跳转到结果页面并传递数据
    return render_template('results.html', result=result)


if __name__ == '__main__':
    # 确保你有一个名为 'templates' 的文件夹，并将 index.html 和 result.html 放在其中
    app.run(debug=True)