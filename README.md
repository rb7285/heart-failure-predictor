# Heart Failure Risk Prediction

This project leverages machine learning to predict the risk of heart failure in patients based on clinical records. The model is built using PyTorch and deployed as an interactive web application using Streamlit.

## 🧠 Model Overview

- **Model Type**: Multi-Layer Perceptron (MLP)
- **Framework**: PyTorch
- **Dataset**: [Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) from the UCI Machine Learning Repository
- **Features**: 12 clinical features including age, ejection fraction, serum creatinine, and more
- **Target**: DEATH_EVENT (binary classification: 0 = survived, 1 = death)

## 🔧 Project Structure

├── heart_failure_clinical_records_dataset.csv # Dataset
├── heart_model.pth # Trained model weights
├── heart_disease.py # Model training script
├── heart_failure_app.py # Streamlit application
├── requirements.txt # Project dependencies
└── README.md # Project documentation

bash
Copy code

## 🚀 How to Run the Streamlit App

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/heart-failure-prediction.git
   cd heart-failure-prediction
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit application:

bash
Copy code
streamlit run heart_failure_app.py
Open your browser and navigate to http://localhost:8501 to interact with the app.

📊 Model Performance
Precision: 53%

Recall: 84%

F1-Score: 65%

ROC-AUC: 0.81

These metrics indicate that the model effectively identifies high-risk patients, which is crucial for timely medical intervention.

📝 Dataset Information

Source: Heart Failure Clinical Records Dataset
Instances: 299
Features: 12
Target: DEATH_EVENT (binary classification)