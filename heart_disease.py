import streamlit as st
import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = MLP(in_features=12)
model.load_state_dict(torch.load("heart_model.pth", map_location=torch.device('cpu')))
model.eval()

st.title("💓 Heart Failure Risk Prediction")
st.write("Enter patient details to predict risk of heart failure.")

age = st.number_input("Age", 20, 100, 50)
ejection_fraction = st.number_input("Ejection Fraction (%)", 10, 80, 35)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.1, 10.0, 1.2)
serum_sodium = st.number_input("Serum Sodium (mEq/L)", 110, 150, 137)
time = st.number_input("Follow-up time (days)", 1, 300, 130)

anaemia = st.selectbox("Anaemia", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
high_blood_pressure = st.selectbox("High Blood Pressure", ["No", "Yes"])
sex = st.selectbox("Sex", ["Male", "Female"])
smoking = st.selectbox("Smoking", ["No", "Yes"])

anaemia = 1 if anaemia=="Yes" else 0
diabetes = 1 if diabetes=="Yes" else 0
high_blood_pressure = 1 if high_blood_pressure=="Yes" else 0
sex = 1 if sex=="Male" else 0
smoking = 1 if smoking=="Yes" else 0

features = np.array([[age, anaemia, diabetes, high_blood_pressure, ejection_fraction,
                      serum_creatinine, serum_sodium, sex, smoking, time, 0, 0]], dtype=np.float32)

features = torch.from_numpy(features)

if st.button("Predict Risk"):
    with torch.no_grad():
        prob = model(features).item()

    pred = 1 if prob >= 0.31 else 0

    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Failure (Risk: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Failure (Risk: {prob*100:.2f}%)")
