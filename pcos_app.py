import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO, StringIO
import datetime

# ------------- Model Loading ----------------
with open("pcos_model.pkl", "rb") as f:
    pcos_model = pickle.load(f)

with open("diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)

# ------------- PDF Report Class ----------------
class PDF(FPDF):
    def header(self):
        self.set_fill_color(0, 102, 204)
        self.set_text_color(255, 255, 255)
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

# ------------- Health Score Logic ----------------
def calculate_health_score(data):
    score = 100
    if data.get("BMI", 0) < 18.5 or data.get("BMI", 0) > 25:
        score -= 15
    if data.get("Waist (cm)", 0) > 88:
        score -= 15
    if data.get("Acne") == "Yes":
        score -= 10
    if data.get("Hair Loss") == "Yes":
        score -= 10

    score = max(0, min(100, score))

    if score >= 80:
        advice = "✅ Excellent! You appear to be in good health."
    elif score >= 50:
        advice = "⚠️ Moderate risk. Consider lifestyle changes and regular checkups."
    else:
        advice = "🚨 High risk. We recommend consulting a gynecologist."

    return score, advice

# ------------- PDF Export Function ----------------
def generate_pdf_report(user_inputs, disease_name, risk_percent, explanation_text, chart_fig=None):
    pdf = PDF()
    pdf.add_page()
    pdf.add_patient_info(user_inputs.get("Name", "Anonymous"), user_inputs.get("Age", "N/A"), user_inputs.get("Gender", "Female"))
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

# Initialize session_state keys if not present
if "last_page" not in st.session_state:
    st.session_state.last_page = "Welcome"
if "tracker_option" not in st.session_state:
    st.session_state.tracker_option = "None"

# Sidebar
st.sidebar.title("🧠 CheckLyf AI")

page = st.sidebar.selectbox("📄 Choose a Page", [
    "Welcome", "PCOS", "Diabetes"
])

# Reset tracker if page is changed
if page != st.session_state.last_page:
    st.session_state.tracker_option = "None"
    st.session_state.last_page = page

tracker_option = st.sidebar.selectbox(
    "🛠️ Choose a Tracker", [
        "None", "🩸 Period Tracker", "🧘 Mood Tracker", "💤 Sleep Tracker"
    ],
    index=["None", "🩸 Period Tracker", "🧘 Mood Tracker", "💤 Sleep Tracker"].index(
        st.session_state.tracker_option
    ),
    key="tracker_option"
)
reminder_enabled = st.sidebar.checkbox("🔔 Enable Daily Log Reminder")
reminder_time = None

if reminder_enabled:
    reminder_time = st.sidebar.time_input("⏰ Choose Reminder Time")


# ------------------ Display Logic ------------------

# Maintain selected tracker across interactions
if "active_tracker" not in st.session_state:
    st.session_state.active_tracker = None

# If user selects a new tracker, update the active one
if tracker_option != st.session_state.active_tracker:
    st.session_state.active_tracker = tracker_option

# Now display based on the active tracker
active_tracker = st.session_state.active_tracker

if active_tracker == "🩸 Period Tracker":
    st.title("🩸 Period Tracker (AI-Powered)")
    st.markdown("Track and understand your menstrual cycle with AI support.")

    period_option = st.selectbox("Choose an option", [
        "Period Date Log",
        "Period Problems Assistant"
    ])

    st.markdown("### 📘 Mood & Symptom Journal")
    mood_today = st.selectbox("How are you feeling today?", ["🙂 Good", "😐 Okay", "🙁 Bad"])
    notes = st.text_area("Anything else you'd like to note?")
    
    if st.button("📝 Save Today's Log"):
        st.success("Log saved! (not stored yet, unless you add storage logic)")


    if period_option == "Period Date Log":
        user_name = st.text_input("Your Name")
        last_period = st.date_input("When did your last period start?", max_value=datetime.date.today())
        cycle_length = st.number_input("Average cycle length (in days)", min_value=20, max_value=45, value=28)
        period_duration = st.number_input("Period duration (in days)", min_value=2, max_value=10, value=5)

        if st.button("🔮 Predict My Next Period"):
            next_period = last_period + datetime.timedelta(days=cycle_length)
            ovulation_day = last_period + datetime.timedelta(days=cycle_length - 14)
            buffer = 2
            estimated_start = next_period - datetime.timedelta(days=buffer)
            estimated_end = next_period + datetime.timedelta(days=buffer)

            if st.button("📤 Export Period Log as PDF"):
                        pdf_data = generate_pdf_report(
                user_inputs={"Name": user_name, "Age": "N/A", "Gender": "Female"},
                disease_name="Menstrual Cycle",
                risk_percent=0.0,
                explanation_text="This report includes your period tracking information.",
                chart_fig=None
            )
            st.download_button("⬇️ Download Period Report", data=pdf_data, file_name="period_report.pdf")


            st.markdown("### 📊 Your Weekly Period Summary")
            st.write("- Period Duration: `{} days`".format(period_duration))
            st.write("- Cycle Length: `{} days`".format(cycle_length))
            st.write("- Last Period Start: `{}`".format(last_period.strftime("%d %b %Y")))

            st.success(f"Hello **{user_name}**, your next period is likely between **{estimated_start.strftime('%d %B')}** and **{estimated_end.strftime('%d %B, %Y')}**")
            st.info(f"🧬 Estimated ovulation day: **{ovulation_day.strftime('%d %B, %Y')}**")
            st.markdown("🛈 _Note: Period dates may vary depending on stress, illness, diet, and hormonal changes._")

            st.markdown("### 🌿 What You Should Eat (AI Suggestion)")
            st.markdown("- 🥦 Iron-rich foods like spinach and lentils")
            st.markdown("- 🍫 Dark chocolate (magnesium)")
            st.markdown("- 🫚 Ginger or chamomile tea for cramps")
            st.markdown("- 🧘‍♀️ Stay hydrated and do light yoga")

            st.markdown("### 📅 Period Timeline Summary")
            st.write(f"- Last period: `{last_period.strftime('%d %b %Y')}`")
            st.write(f"- Predicted range: `{estimated_start.strftime('%d %b')} – {estimated_end.strftime('%d %b %Y')}`")
            st.write(f"- Period duration: `{period_duration} days`")
            st.write(f"- Cycle length: `{cycle_length} days`")

    elif period_option == "Period Problems Assistant":
        st.subheader("🤖 Period Problems Assistant")
        user_question = st.text_input("Ask me anything about your period problems:")

        if user_question.strip() != "":
            q = user_question.lower()

            if "cramps" in q and "after" in q:
                st.markdown("Cramps after your period can be due to ovulation, hormonal changes, or underlying conditions like endometriosis. If pain is severe or frequent, consult a doctor.")
            elif "late" in q or "delay" in q:
                st.markdown("Periods can be delayed due to stress, diet, travel, or medical conditions like PCOS or thyroid imbalance.")
            elif "irregular" in q:
                st.markdown("Irregular periods can result from PCOS, stress, or lifestyle changes. Keeping a healthy routine helps.")
            elif "spotting" in q:
                st.markdown("Spotting between periods could be due to ovulation, hormonal imbalance, or contraception. If it continues, talk to a doctor.")
            elif "heavy" in q:
                st.markdown("Heavy periods might be caused by fibroids, hormonal imbalances, or other conditions. Keep a record and speak with your doctor.")
            elif "missed" in q:
                st.markdown("A missed period isn't always pregnancy—it can be due to stress, diet, or hormonal imbalance.")
            elif "painful" in q or "dysmenorrhea" in q:
                st.markdown("Painful periods (dysmenorrhea) can be due to high prostaglandins or conditions like endometriosis.")
            elif "normal cycle" in q or "cycle length" in q:
                st.markdown("A normal menstrual cycle lasts between 21 to 35 days, and bleeding lasts 2 to 7 days.")
            elif "breast" in q and "pain" in q:
                st.markdown("Breast pain before periods is caused by hormonal changes, especially rising estrogen and progesterone. These hormones make the breast ducts enlarge and retain water, causing soreness.")
            elif "bloat" in q or ("stomach" in q and "bloat" in q):
                st.markdown("Bloating before your period is due to hormonal changes that cause your body to retain water and salt. It’s common in PMS and usually reduces once bleeding starts.")
            elif "brown discharge" in q:
                st.markdown("Brown discharge before or after a period is usually old blood leaving the uterus. It’s common and usually harmless.")
            elif "white discharge" in q:
                st.markdown("White discharge before your period is normal and helps keep your vagina clean. If it's smelly or itchy, it might indicate an infection.")
            elif "period twice" in q or "2 periods" in q:
                st.markdown("Two periods in one month can happen due to stress, hormonal imbalance, or birth control. If it happens often, see a doctor.")
            elif "light period" in q:
                st.markdown("A light period may be caused by stress, diet, weight changes, or birth control. It’s usually okay unless it continues.")
            elif "clots" in q or "blood clots" in q:
                st.markdown("Blood clots during periods are normal if small. Large clots or heavy bleeding could signal a medical issue like fibroids.")
            elif "period lasts" in q:
                st.markdown("A normal period lasts 2 to 7 days. If it’s shorter or longer regularly, it’s best to get checked.")
            elif "period pain remedy" in q or "cramp relief" in q:
                st.markdown("Use a heating pad, stay hydrated, take light walks, or try ibuprofen. Severe pain? Visit a doctor.")
            elif "period not stopping" in q or "bleeding won't stop" in q:
                st.markdown("Prolonged bleeding can be caused by hormonal imbalance, fibroids, or other conditions. Please consult a gynecologist.")
            elif "period during pregnancy" in q:
                st.markdown("True periods don't occur during pregnancy, but light spotting might. Always consult a doctor to confirm.")
            elif "period after sex" in q:
                st.markdown("Period-like bleeding after sex may be from cervical irritation or hormonal shifts. If it continues, get it checked.")
            else:
                st.markdown("I’m still learning! For serious or persistent issues, it’s always best to consult a gynecologist.")

elif active_tracker == "🧘 Mood Tracker":
    st.title("🧘 Mood Tracker (Coming Soon)")
    st.info("This tracker is under development.")

elif active_tracker == "💤 Sleep Tracker":
    st.title("💤 Sleep Tracker (Coming Soon)")
    st.info("This tracker is under development.")


# Only show disease pages if no tracker is selected
elif tracker_option == "None":
    if page == "Welcome":
        st.title("🏥 Welcome to CheckLyf AI")
        st.write("Your AI-based health assistant.")
        st.markdown("- ✅ Predict diseases like PCOS and Diabetes")
        st.markdown("- 📊 Get reports and insights instantly")

    elif page == "PCOS":
        st.title("🌧️ PCOS Risk Checker")
        st.markdown("## 📝 Enter Your Information")

        user_name = st.text_input("📝 Enter your full name", "")
        age = st.slider("🎂 Age", 15, 45, 25)
        weight = st.number_input("⚖️ Weight (kg)", 30.0, 120.0, 65.0)
        height = st.number_input("📏 Height (cm)", 140.0, 190.0, 160.0)
        waist = st.number_input("👖 Waist (cm)", 50.0, 120.0, 80.0)
        acne = st.selectbox("🪕 Do you have acne?", ["No", "Yes"])
        hair_loss = st.selectbox("💇‍♀️ Do you suffer from hair loss?", ["No", "Yes"])

        if st.button("🧬 Predict PCOS Risk"):
            if user_name.strip() == "":
                st.warning("⚠️ Please enter your name before proceeding.")
            else:
                acne_binary = 1 if acne == "Yes" else 0
                hair_loss_binary = 1 if hair_loss == "Yes" else 0
                bmi = weight / ((height / 100) ** 2)

                input_data = np.array([[age, weight, height, waist, acne_binary, hair_loss_binary]])
                prediction = pcos_model.predict_proba(input_data)[0][1]
                pcos_risk_percent = round(prediction * 100, 2)

                st.success(f"Your estimated PCOS risk is **{pcos_risk_percent}%**")

                inputs = {
                    "Name": user_name,
                    "Age": age,
                    "Weight (kg)": weight,
                    "Height (cm)": height,
                    "Waist (cm)": waist,
                    "Acne": acne,
                    "Hair Loss": hair_loss,
                    "BMI": round(bmi, 2)
                }

                score, advice = calculate_health_score(inputs)
                st.markdown(f"### 💡 Your Health Score: `{score}/100`")
                st.info(f"🩺 Personalized Advice: {advice}")

                fig, ax = plt.subplots()
                ax.bar(["You", "Max"], [pcos_risk_percent, 100], color=["orange", "lightgray"])
                ax.set_ylabel("Risk (%)")
                ax.set_title("Your PCOS Risk Level")
                st.pyplot(fig)

                explanation = (
                    "AI assessed your risk using BMI, waist size, acne, and hair loss. "
                    "These are common indicators in clinical PCOS assessment."
                )

                pdf_buffer = generate_pdf_report(inputs, "PCOS", pcos_risk_percent, explanation, fig)
                st.download_button("📥 Download Report (PDF)", data=pdf_buffer, file_name="pcos_checklyf_report.pdf")

        st.markdown("---")
        st.markdown("### 📁 Or Upload a CSV File")
        uploaded_file = st.file_uploader("Upload a PCOS data CSV (optional)", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                st.write("📊 Preview of uploaded data:")
                st.dataframe(df)

                st.markdown("### 🔍 Prediction Results from CSV:")
                for index, row in df.iterrows():
                    age = row.get("Age", 25)
                    weight = row.get("Weight (kg)", 65)
                    height = row.get("Height (cm)", 160)
                    waist = row.get("Waist (cm)", 80)
                    acne = 1 if row.get("Acne", "No") == "Yes" else 0
                    hair_loss = 1 if row.get("Hair Loss", "No") == "Yes" else 0
                    bmi = weight / ((height / 100) ** 2)

                    input_data = np.array([[age, weight, height, waist, acne, hair_loss]])
                    prediction = pcos_model.predict_proba(input_data)[0][1]
                    risk_percent = round(prediction * 100, 2)

                    st.write(f"🧬 Row {index + 1} — Risk: **{risk_percent}%**")

            except Exception as e:
                st.error(f"⚠️ Error reading file: {e}")

    elif page == "Diabetes":
        st.title("🩸 Diabetes Risk Checker")

        pregnancies = st.slider("🤰 Number of Pregnancies", 0, 15, 1)
        age = st.slider("🎂 Age", 10, 90, 30)

        q1 = st.selectbox("😓 Do you feel tired or thirsty frequently?", ["No", "Yes"])
        q2 = st.selectbox("💉 Do you have high blood pressure?", ["No", "Yes"])
        q3 = st.selectbox("🟤 Do you have dark patches of skin?", ["No", "Yes"])
        q4 = st.selectbox("⚖️ Have you experienced sudden weight gain?", ["No", "Yes"])
        q5 = st.selectbox("🏋️ Are you overweight?", ["No", "Yes"])
        q6 = st.selectbox("👨‍👩‍👧‍👦 Any family history of diabetes?", ["No", "Yes"])

        user_name = st.text_input("📝 Enter your full name", "")

        if st.button("🔍 Predict Diabetes Risk"):
            if user_name.strip() == "":
                st.warning("⚠️ Please enter your name before proceeding.")
            else:
                glucose = 160 if q1 == "Yes" else 110
                bp = 130 if q2 == "Yes" else 80
                skin = 35 if q3 == "Yes" else 20
                insulin = 300 if q4 == "Yes" else 100
                bmi = 32 if q5 == "Yes" else 24
                dpf = 0.9 if q6 == "Yes" else 0.3

                input_data = pd.DataFrame([{
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": bp,
                    "SkinThickness": skin,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age
                }])

                prob = diabetes_model.predict_proba(input_data)[0]
                risk_percent = round(prob[1] * 100, 2)

                if risk_percent > 50:
                    st.error(f"⚠️ {user_name}, High Risk Detected: {risk_percent}%")
                else:
                    st.success(f"🎉 {user_name}, Low Risk: {risk_percent}%")

                fig, ax = plt.subplots(figsize=(6, 1.2))
                ax.barh(["Risk"], [risk_percent], color="salmon" if risk_percent > 50 else "lightgreen")
                ax.set_xlim(0, 100)
                ax.set_title("Diabetes Risk Level")
                st.pyplot(fig)

                diabetes_inputs = {
                    "Name": user_name,
                    "Pregnancies": pregnancies,
                    "Age": age,
                    "Tired/Thirsty": q1,
                    "BP Issues": q2,
                    "Dark Skin Patches": q3,
                    "Sudden Weight Gain": q4,
                    "Overweight": q5,
                    "Family History": q6,
                }

                explanation = "AI prediction based on known diabetes risk factors like BMI, glucose levels, and family history."
                pdf_buffer = generate_pdf_report(diabetes_inputs, "Diabetes", risk_percent, explanation, fig)

                st.download_button("📥 Download Report (PDF)", data=pdf_buffer, file_name="diabetes_checklyf_report.pdf")

                st.markdown("## 📁 Optional: Upload a CSV file")
                uploaded_file = st.file_uploader("Upload your diabetes data (CSV)", type=["csv"])

                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.success("✅ File uploaded successfully!")
                        st.write("📊 Uploaded Data Preview:")
                        st.dataframe(df.head())
                        st.warning("⚠️ Automatic prediction from CSV not yet enabled. Please use manual form above.")
                    except Exception as e:
                        st.error(f"❌ Error reading CSV: {e}")
