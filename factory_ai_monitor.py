"""
This script creates a Streamlit web application for monitoring sensor data in a factory.

It uses a local instance of the Qwen1.5-1.8B-Chat model to analyze sensor data,
detect anomalies, and provide explanations and recommendations in both Chinese and
English.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 1. Load Local LLM ==========
@st.cache_resource
def load_local_llm() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Loads the local Qwen1.5-1.8B-Chat model and tokenizer.

    Returns:
        A tuple containing the tokenizer and the model.
    """
    model_name = "Qwen/Qwen1.5-1.8B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, model

tokenizer, model = load_local_llm()

# ========== 2. Language Selection ==========
lang = st.sidebar.radio("选择语言 / Select Language", ["中文", "English"])

# ========== 3. Sensor Data Simulation ==========
@st.cache_data
def generate_sensor_data(n: int = 100, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a DataFrame of simulated sensor data.

    Args:
        n: The number of data points to generate.
        random_seed: The random seed to use for data generation.

    Returns:
        A DataFrame of simulated sensor data.
    """
    np.random.seed(random_seed)
    data = []
    for i in range(n):
        row = {
            'timestamp': pd.Timestamp.now() + pd.Timedelta(seconds=i * 2),
            'temperature': np.random.normal(28, 1.5),
            'humidity': np.random.normal(55, 2),
            'vibration': np.random.normal(0.5, 0.1),
            'power': np.random.normal(110, 5),
            'pressure': np.random.normal(2.0, 0.05)
        }
        # Add random anomalies
        if np.random.rand() < 0.06:
            row['temperature'] += np.random.uniform(6, 12)
        if np.random.rand() < 0.05:
            row['vibration'] += np.random.uniform(0.7, 1.3)
        if np.random.rand() < 0.05:
            row['power'] += np.random.uniform(20, 40)
        data.append(row)
    return pd.DataFrame(data)

def detect_anomalies(df: pd.DataFrame, threshold: int = 3) -> pd.Series:
    """
    Detects anomalies in the sensor data.

    Args:
        df: The DataFrame of sensor data.
        threshold: The z-score threshold for anomaly detection.

    Returns:
        A Series of booleans indicating whether each data point is an anomaly.
    """
    selected = ['temperature', 'humidity', 'vibration', 'power', 'pressure']
    z_scores = (df[selected] - df[selected].mean()) / df[selected].std()
    anomalies = (np.abs(z_scores) > threshold)
    return anomalies.any(axis=1)

# ========== 4. Multi-language Anomaly Explanation ==========
def qwen_anomaly_explain(row: pd.Series, lang: str = "中文") -> str:
    """
    Generates an explanation for an anomaly using the local LLM.

    Args:
        row: The data point to explain.
        lang: The language to use for the explanation.

    Returns:
        The explanation for the anomaly.
    """
    if lang == "中文":
        prompt = f"""
        工厂传感器实时监控数据如下：
        时间：{row['timestamp']}
        温度：{row['temperature']:.2f}℃
        湿度：{row['humidity']:.2f}%
        振动：{row['vibration']:.2f} g
        功率：{row['power']:.2f} kW
        压力：{row['pressure']:.2f} bar
        请用专业工业人工智能角度，判断该数据异常的潜在原因和建议措施，不少于50字。
        """
    else:
        prompt = f"""
        Factory sensor monitoring data:
        Time: {row['timestamp']}
        Temperature: {row['temperature']:.2f}℃
        Humidity: {row['humidity']:.2f}%
        Vibration: {row['vibration']:.2f} g
        Power: {row['power']:.2f} kW
        Pressure: {row['pressure']:.2f} bar
        As an AI expert for smart factories, please analyze the possible causes of this
        anomaly and provide professional suggestions (at least 50 words).
        """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.replace(prompt, "").strip()

# ========== 5. Multi-language UI ==========
if lang == "中文":
    st.title("AI驱动的涡流纺精密控制仿真系统 (AI-Driven Precision Control in Vortex Spinning)")
    st.markdown("### 智能工厂本地大模型异常监控仿真系统")
    st.write("本系统模拟纺织工厂实时传感器数据，自动检测异常并用本地大语言模型分析异常原因及建议。")
    st.sidebar.header("仿真参数设置")
    data_expander_label = "全部传感器数据"
    anomaly_title = "检测到的异常数据点（可点按钮AI本地解释）"
    no_anomaly_info = "未检测到明显异常数据。可调整点数或种子。"
    btn_label = "用本地大模型AI分析异常"
    running_msg = "大模型推理中..."
else:
    st.title("AI-Driven Precision Control in Vortex Spinning")
    st.markdown("### Smart Factory Local LLM Anomaly Monitoring Demo")
    st.write(
        "This system simulates real-time sensor data from a textile factory, "
        "detects anomalies, and uses a local LLM for AI-driven explanations "
        "and recommendations."
    )
    st.sidebar.header("Simulation Settings")
    data_expander_label = "All Sensor Data"
    anomaly_title = "Detected Anomalies (Click Button for LLM Explanation)"
    no_anomaly_info = "No significant anomalies detected. Try increasing points or changing the seed."
    btn_label = "AI Analyze Anomaly with Local LLM"
    running_msg = "Running local LLM inference..."

# ========== 6. Main ==========
def main():
    """
    The main function of the Streamlit application.
    """
    num_points = st.sidebar.slider("传感器采样点数 / Sensor Points", 50, 200, 60)
    seed = st.sidebar.number_input("随机种子 / Random Seed", 1, 99999, 42)

    with st.spinner("正在生成传感器数据并检测异常..." if lang == "中文" else "Generating sensor data and detecting anomalies..."):
        df = generate_sensor_data(num_points, seed)
        df['is_anomaly'] = detect_anomalies(df)
        anomaly_df = df[df['is_anomaly']]

    st.subheader("传感器趋势 (温度、功率、振动)" if lang == "中文" else "Sensor Trends (Temperature, Power, Vibration)")
    fig, ax = plt.subplots()
    df.set_index('timestamp')[['temperature', 'power', 'vibration']].plot(ax=ax)
    st.pyplot(fig)

    with st.expander(data_expander_label):
        st.dataframe(df, use_container_width=True)

    st.subheader(anomaly_title)
    if not anomaly_df.empty:
        for i, row in anomaly_df.iterrows():
            expander_title = (
                f"[{row['timestamp']}] 异常 - 温度: {row['temperature']:.1f}℃  "
                f"振动: {row['vibration']:.2f}g  功率: {row['power']:.2f}kW"
                if lang == "中文"
                else (
                    f"[{row['timestamp']}] Anomaly - Temperature: {row['temperature']:.1f}℃  "
                    f"Vibration: {row['vibration']:.2f}g  Power: {row['power']:.2f}kW"
                )
            )
            with st.expander(expander_title):
                st.write(
                    f"湿度: {row['humidity']:.2f}% | 压力: {row['pressure']:.2f} bar"
                    if lang == "中文"
                    else f"Humidity: {row['humidity']:.2f}% | Pressure: {row['pressure']:.2f} bar"
                )
                if st.button(f"{btn_label} ({i+1})", key=f"ai_explain_{i}"):
                    with st.spinner(running_msg):
                        explanation = qwen_anomaly_explain(row, lang=lang)
                    st.success(explanation)
    else:
        st.info(no_anomaly_info)

    st.caption("© Liao Hu + HuggingFace Transformers | Qwen1.5-1.8B-Chat本地模型 Demo")

if __name__ == "__main__":
    main()