# LLM Factory Demo

This project is a demonstration of a local LLM-powered factory monitoring system. It uses a local instance of the Qwen1.5-1.8B-Chat model to analyze sensor data, detect anomalies, and provide explanations and recommendations in both Chinese and English.

## Features

*   **Local LLM:** Uses a local instance of the Qwen1.5-1.8B-Chat model for anomaly detection and explanation.
*   **Anomaly Detection:** Automatically detects anomalies in sensor data based on a set of predefined rules.
*   **Multi-language Support:** Provides explanations and recommendations in both Chinese and English.
*   **Streamlit Interface:** Uses Streamlit to create an interactive web interface for visualizing sensor data and anomaly detection results.

## How to Run

1.  Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
2.  Run the Streamlit application:
    ```
    streamlit run factory_ai_monitor.py
    ```

## Credits

*   **Liao Hu:** The author of this project.
*   **Hugging Face:** For the Transformers library and the Qwen1.5-1.8B-Chat model.