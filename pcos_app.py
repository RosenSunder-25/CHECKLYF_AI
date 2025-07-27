import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("pcos_model.pkl", "rb"))

# Set page config
st.set_page_config(page_title="🧬 PCOS Detector", layout="centered")

# Title
st.title("👩‍⚕️ PCOS Prediction App")
st.markdown("Get a quick, AI-powered estimate for PCOS based on common symptoms.")

# Input fields
age = st.slider("Age", 15, 45, 25)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=120.0, value=65.0)
height = st.number_input("Height (cm)", min_value=140.0, max_value=190.0, value=160.0)
waist = st.number_input("Waist (cm)", min_value=50.0, max_value=120.0, value=80.0)
acne = st.selectbox("Do you have acne?", ["No", "Yes"])
hair_loss = st.selectbox("Do you suffer from hair loss?", ["No", "Yes"])

st.markdown("## 📤 Or upload CSV for batch prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        X = df[["Age", "Weight", "Height", "Waist", "Acne", "HairLoss"]]
        preds = model.predict_proba(X)
        df["PCOS Probability (%)"] = (preds[:,1] * 100).round(2)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error in file format: {e}")

# Convert categorical to numeric
acne = 1 if acne == "Yes" else 0
hair_loss = 1 if hair_loss == "Yes" else 0

# Predict
if st.button("🔍 Predict"):
    input_data = np.array([[age, weight, height, waist, acne, hair_loss]])
    result = model.predict_proba(input_data)[0]

    pcos_prob = result[1] * 100 # Probability of PCOS
    no_pcos_prob = result[0] * 100

    # Emoji result
    if pcos_prob > 60:
        st.error(f"⚠️ There's a {pcos_prob:.2f}% chance you may have PCOS.\nPlease consult a doctor.")
    else:
        st.success(f"🎉 You're at low risk! Only {pcos_prob:.2f}% chance of PCOS.")

    # Show Pie Chart
    fig, ax = plt.subplots()
    ax.pie([pcos_prob, no_pcos_prob], labels=["PCOS Risk", "No PCOS"], colors=["#FF9999", "#99FF99"],
           autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis("equal")
    st.pyplot(fig)