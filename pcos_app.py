import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
from fpdf import FPDF 
from io import BytesIO 
import datetime
import os 
import base64

# Set default page
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ---------- Load PCOS model ----------
with open("pcos_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------- PDF Generator Class ----------
class PDF(FPDF):
    def header(self):
        self.set_fill_color(0, 102, 204) # ✅ Correct method
        self.set_text_color(255, 255, 255) # ✅ Correct method
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "CheckLyf AI Health Report", ln=True, align="C", fill=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_text_color(169, 169, 169)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Report generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'C')

    def add_patient_info(self, name, age, gender):
        self.set_text_color(0, 0, 0)
        self.set_font("Arial", "", 12)
        self.cell(0, 10, f"Name: {name}", ln=True)
        self.cell(0, 10, f"Age: {age}", ln=True)
        self.cell(0, 10, f"Gender: {gender}", ln=True)
        self.ln(5)

    def add_report_section(self, title, result, features_dict, explanation):
        self.set_fill_color(220, 220, 220)
        self.set_text_color(0, 0, 0)
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, ln=True, fill=True)
        self.set_font("Arial", "", 12)
        self.cell(0, 10, f"Prediction: {result}", ln=True)
        for key, value in features_dict.items():
            self.cell(0, 10, f"{key}: {value}", ln=True)
        self.multi_cell(0, 10, f"Explanation: {explanation}")
        self.ln(10)

    def add_footer_disclaimer(self):
        self.set_y(-30)
        self.set_font("Arial", "I", 10)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 10, "Disclaimer: This is an AI-generated prediction. Please consult a doctor for medical advice.")

# ---------- PDF Report Generation Function ----------
def generate_pdf_report(user_inputs, disease_name, risk_percent, explanation_text, chart_fig=None):
    pdf = PDF()
    pdf.add_page()
    pdf.add_patient_info("Anonymous", user_inputs.get("Age", "N/A"), user_inputs.get("Gender", "Female"))
    pdf.add_report_section(
        title=f"{disease_name} Risk Analysis",
        result=f"{risk_percent:.2f}% Risk",
        features_dict=user_inputs,
        explanation=explanation_text
    )

    if chart_fig:
        img_buf = BytesIO()
        chart_fig.savefig(img_buf, format="PNG")
        img_buf.seek(0)
        pdf.image(img_buf, x=10, y=pdf.get_y(), w=pdf.w - 20)
        img_buf.close()

    pdf.add_footer_disclaimer()
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# ---------- Sidebar Navigation ----------
st.sidebar.title("🧠 CheckLyf AI")
disease = st.sidebar.selectbox("Choose a disease", ["Welcome", "PCOS", "Diabetes"])

if "page" not in st.session_state:
    st.session_state.page = "welcome"

if disease == "Welcome":
    st.session_state.page = "welcome"
elif disease == "PCOS":
    st.session_state.page = "pcos"
elif disease == "Diabetes":
    st.session_state.page = "diabetes"

# ---------- Welcome Page ----------
if st.session_state.page == "welcome":
    st.title("🏥 Welcome to CheckLyf AI")
    st.markdown("""
        #### Your AI-based health assistant  
        - Predict diseases like **PCOS** and **Diabetes**  
        - Get reports and insights instantly  
    """)

    if st.button("Start PCOS Prediction"):
        st.session_state.page = "pcos"
        st.rerun()


# ---------- PCOS Page ----------
elif st.session_state.page == "pcos":
    st.title("🌧️ PCOS Risk Checker")

    st.markdown("## 📝 Enter Your Information")

    # Manual Input
    age = st.slider("🎂 Age", 15, 45, 25)
    weight = st.number_input("⚖️ Weight (kg)", 30.0, 120.0, 65.0)
    height = st.number_input("📏 Height (cm)", 140.0, 190.0, 160.0)
    waist = st.number_input("👖 Waist (cm)", 50.0, 120.0, 80.0)
    acne = st.selectbox("🪕 Do you have acne?", ["No", "Yes"])
    hair_loss = st.selectbox("💇‍♀️ Do you suffer from hair loss?", ["No", "Yes"])

    acne_binary = 1 if acne == "Yes" else 0
    hair_loss_binary = 1 if hair_loss == "Yes" else 0

    if st.button("🧬 Predict PCOS Risk"):
        bmi = weight / ((height / 100) ** 2)
        input_data = np.array([[age, weight, height, waist, acne_binary, hair_loss_binary]])
        prediction = model.predict_proba(input_data)[0][1]
        pcos_risk_percent = round(prediction * 100, 2)

        st.success(f"Your estimated PCOS risk is **{pcos_risk_percent}%**")

        # Chart
        st.markdown("## 📈 Risk Probability Graph")
        fig, ax = plt.subplots()
        ax.bar(["You", "Max"], [pcos_risk_percent, 100], color=["orange", "lightgray"])
        ax.set_ylabel("Risk (%)")
        ax.set_title("Your PCOS Risk Level")
        st.pyplot(fig)

        # Info
        inputs = {
            "Age": age,
            "Weight (kg)": weight,
            "Height (cm)": height,
            "Waist (cm)": waist,
            "Acne": acne,
            "Hair Loss": hair_loss,
            "BMI": round(bmi, 2)
        }

        explanation = (
            "AI assessed your risk using BMI, waist size, acne, and hair loss. "
            "These are common indicators in clinical PCOS assessment."
        )

        # PDF
        pdf_buffer = generate_pdf_report(
            user_inputs=inputs,
            disease_name="PCOS",
            risk_percent=pcos_risk_percent,
            explanation_text=explanation,
            chart_fig=fig
        )

        st.markdown("## 🧾 Professional PDF Report")
        st.download_button(
            "📅 Download PCOS Report (PDF)",
            data=pdf_buffer,
            file_name="pcos_checklyf_report.pdf",
            mime="application/pdf"
        )

        # TXT download
        summary_txt = f"PCOS Risk Report\n\nRisk: {pcos_risk_percent}%\n\nInputs:\n" + "\n".join(
            f"{key}: {value}" for key, value in inputs.items()
        )
        st.download_button(
            label="📄 Download Summary (.txt)",
            data=summary_txt,
            file_name="pcos_summary.txt",
            mime="text/plain"
        )

    # Optional: CSV Bulk Prediction
    st.markdown("## 📂 Upload CSV to Predict in Bulk (Optional)")
    csv_file = st.file_uploader("Upload CSV with columns: Age,Weight,Height,Waist,Acne,Hair_Loss", type=["csv"])

    if csv_file:
        df = pd.read_csv(csv_file)
        df["Acne"] = df["Acne"].map({"Yes": 1, "No": 0})
        df["Hair_Loss"] = df["Hair_Loss"].map({"Yes": 1, "No": 0})
        X = df[["Age", "Weight", "Height", "Waist", "Acne", "Hair_Loss"]]
        df["PCOS Risk %"] = model.predict_proba(X)[:, 1] * 100
        st.dataframe(df)

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Bulk Predictions (CSV)",
            data=csv_out,
            file_name="pcos_bulk_predictions.csv",
            mime="text/csv"
        )





# --- Diabetes Prediction Page ---
if st.session_state.page == "diabetes":
    st.title("🩸 Diabetes Risk Checker (Simple Mode)")
    st.markdown("Answer a few health-related questions to estimate your diabetes risk.")

    pregnancies = st.slider("🤰 How many times have you been pregnant?", 0, 15, 1)
    age = st.slider("🎂 Your age", 18, 80, 30)

    q1 = st.radio("💧 Do you often feel tired or thirsty?", ["Yes", "No"])
    q2 = st.radio("💓 Do you have blood pressure issues?", ["Yes", "No"])
    q3 = st.radio("🟤 Do you notice dark patches on your skin?", ["Yes", "No"])
    q4 = st.radio("📈 Have you gained weight suddenly?", ["Yes", "No"])
    q5 = st.radio("⚖️ Are you overweight or obese?", ["Yes", "No"])
    q6 = st.radio("👨‍👩‍👧‍👦 Do any family members have diabetes?", ["Yes", "No"])

    st.markdown("### 📁 Or Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Drag and drop your CSV file here", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        with open("diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
        preds = model.predict(df)
        st.write("🔎 **Predictions:**")
        st.write(preds)

    if st.button("🔍 Predict Diabetes (Based on answers above)"):
        glucose = 160 if q1 == "Yes" else 110
        bp = 130 if q2 == "Yes" else 80
        skin = 35 if q3 == "Yes" else 20
        insulin = 300 if q4 == "Yes" else 100
        bmi = 32 if q5 == "Yes" else 24
        dpf = 0.9 if q6 == "Yes" else 0.3

        input_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]
        with open("diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
        prob = model.predict_proba(input_data)[0]

        risk_percent = round(prob[1] * 100, 2)
        st.session_state["diabetes_risk"] = risk_percent

        if risk_percent > 50:
            st.error(f"⚠️ You may be at risk. Probability: {risk_percent:.2f}%")
        else:
            st.success(f"🎉 You're at low risk! Probability: {risk_percent:.2f}%")

        # 📊 Risk Graph
        st.markdown("### 📊 Risk Probability Graph")
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh(["Risk"], [risk_percent], color="salmon" if risk_percent > 50 else "lightgreen")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Diabetes Risk (%)")
        ax.set_title("Diabetes Risk Level")
        st.pyplot(fig)

        st.session_state["diabetes_fig"] = fig # Save for report

        # 📋 Prepare Inputs for Report
        diabetes_inputs = {
            "Pregnancies": pregnancies,
            "Age": age,
            "Tired/Thirsty": q1,
            "BP Issues": q2,
            "Dark Skin Patches": q3,
            "Sudden Weight Gain": q4,
            "Overweight": q5,
            "Family History": q6,
        }

        explanation = (
            "AI estimated your risk based on input features like glucose, blood pressure, insulin levels, "
            "BMI, and family history. These are clinically associated with the development of Type 2 Diabetes."
        )

        # --- Generate PDF ---
        pdf_buffer = generate_pdf_report(
            user_inputs=diabetes_inputs,
            disease_name="Diabetes",
            risk_percent=risk_percent,
            explanation_text=explanation,
            chart_fig=fig
        )

        st.markdown("### 🧾 Professional PDF Report")
        st.download_button(
            "📥 Download Diabetes Report (PDF)",
            data=pdf_buffer,
            file_name="diabetes_checklyf_report.pdf",
            mime="application/pdf"
        )

        # --- Plain Text Report ---
        from io import StringIO
        import datetime

        text_report = StringIO()
        text_report.write("🧾 CheckLyf AI - Diabetes Report\n")
        text_report.write("=" * 40 + "\n")
        text_report.write(f"📅 Date: {datetime.date.today()}\n\n")
        text_report.write("👤 User Details:\n")
        text_report.write(f"Pregnancies: {pregnancies}\n")
        text_report.write(f"Age: {age}\n\n")
        text_report.write("📊 Health Responses:\n")
        text_report.write(f"Tired/Thirsty: {q1}\n")
        text_report.write(f"Blood Pressure Issues: {q2}\n")
        text_report.write(f"Dark Skin Patches: {q3}\n")
        text_report.write(f"Sudden Weight Gain: {q4}\n")
        text_report.write(f"Overweight/Obese: {q5}\n")
        text_report.write(f"Family History: {q6}\n\n")
        text_report.write("🤖 AI Analysis:\n")
        text_report.write(f"Predicted Risk: {risk_percent:.2f}%\n")
        text_report.write("Recommendation: " + (
            "Consult a doctor soon. Risk appears high." if risk_percent > 50 else "Maintain a healthy lifestyle."
        ) + "\n")

        st.download_button(
            label="📥 Download Report (.txt)",
            data=text_report.getvalue(),
            file_name="diabetes_risk_report.txt",
            mime="text/plain"
        )

