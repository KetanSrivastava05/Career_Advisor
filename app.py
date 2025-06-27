import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from langchain_together import ChatTogether

# Load model and scaler
model = load_model("salary_predictor_model.keras")
scaler = joblib.load("scaler1.save")

# Field of study list
field_list = ['Business', 'Computer Science', 'Engineering', 'Law', 'Mathematics', 'Medicine']

# Weight functions
def weight_projects(val):
    if val <= 2: return val * 1.5
    elif val <= 5: return val * 2
    elif val <= 7: return val * 2.5
    else: return val * 3

def weight_internships(val):
    if val == 0: return val * 1.2
    elif val == 1: return val * 1.8
    elif val == 2: return val * 2.2
    else: return val * 2.8

def weight_certifications(val):
    if val <= 1: return val * 1.1
    elif val <= 3: return val * 1.6
    else: return val * 2.1

def weight_soft_skills(val):
    if val <= 3: return val * 1
    elif val <= 6: return val * 1.5
    elif val <= 8: return val * 2
    else: return val * 2.5

def weight_uni_gpa(val):
    if val < 7: return val * 1.2
    elif val <= 8.5: return val * 1.6
    elif val <= 9.2: return val * 2.0
    else: return val * 2.4

def weight_hs_per(val):
    if val < 60: return val * 1.1
    elif val <= 75: return val * 1.5
    elif val <= 90: return val * 1.8
    else: return val * 2.2

# Title and instructions
st.set_page_config(page_title="K10 Career Predictor", layout="centered")
st.title("\U0001F4BC K10 Career Predictor")
st.markdown("""
Estimate your expected salary range and get personalized career skill suggestions based on your academic profile.

---
""")

# Input fields
col1, col2 = st.columns(2)

with col1:
    projects = st.number_input("Projects Completed", min_value=0, value=2)
    certifications = st.number_input("Certifications Done", min_value=0, value=2)
    hs_per = st.slider("High School %", 0.0, 100.0, value=80.0)
    field_study_input = st.selectbox("Field of Study", field_list, index=1)

with col2:
    internships = st.number_input("Internships Completed", min_value=0, value=1)
    soft_skills = st.slider("Soft Skills Score (0–10)", 0.0, 10.0, value=6.0)
    uni_gpa = st.slider("University GPA (out of 10)", 0.0, 10.0, value=7.5)
    target_salary = st.number_input("Target Salary (in ₹)", min_value=0, value=600000)

if st.button("Predict and Get Advice"):
    try:
        # Field encoding
        field_encoding = {
            f"Field_of_Study_{field}": int(field == field_study_input)
            for field in field_list
        }

        # Weights
        w_projects = weight_projects(projects)
        w_internships = weight_internships(internships)
        w_certifications = weight_certifications(certifications)
        w_soft_skills = weight_soft_skills(soft_skills)
        w_hs_per = weight_hs_per(hs_per)
        w_uni_gpa = weight_uni_gpa(uni_gpa)

        gpa_x_projects = w_uni_gpa * projects
        gpa_x_internships = w_uni_gpa * internships

        input_df = pd.DataFrame([{
            'w_Projects_Completed': w_projects,
            'w_Internships_Completed': w_internships,
            'w_Certifications': w_certifications,
            'w_Soft_Skills_Score': w_soft_skills,
            'w_High_School_Per': w_hs_per,
            'w_University_GPA': w_uni_gpa,
            'GPA_x_Projects': gpa_x_projects,
            'GPA_x_Internships': gpa_x_internships,
            **field_encoding
        }])

        input_scaled = scaler.transform(input_df)
        predicted_salary = model.predict(input_scaled)[0][0]
        ps1 = predicted_salary * 0.3
        ps2 = predicted_salary * 0.5

        # Strengths & Weaknesses
        weak_areas, strength_areas = [], []
        if projects < 3: weak_areas.append("limited project experience")
        else: strength_areas.append("good project experience")
        if internships < 3: weak_areas.append("low industry exposure")
        else: strength_areas.append("strong industry exposure")
        if certifications < 3: weak_areas.append("few certifications")
        else: strength_areas.append("diverse technical certifications")
        if soft_skills < 7: weak_areas.append("soft skills need improvement")
        else: strength_areas.append("strong interpersonal skills")
        if hs_per < 85: weak_areas.append("average high school academics")
        else: strength_areas.append("solid high school academics")
        if uni_gpa < 8: weak_areas.append("moderate university GPA")
        else: strength_areas.append("high university GPA")

        strength_summary = " and ".join(strength_areas)
        weakness_summary = " and ".join(weak_areas)

        user_context = f"""
The user has the following profile:
- Projects: {projects}
- Internships: {internships}
- Certifications: {certifications}
- Soft Skills: {soft_skills}
- High School %: {hs_per}
- University GPA: {uni_gpa}
- Field: {field_study_input}
Strengths: {strength_summary}
Weaknesses: {weakness_summary}
"""


        if target_salary > ps2:
            prompt = f"""{user_context}
The user's target salary ₹{target_salary:,.2f} is **higher than the expected range** of ₹{ps1:,.0f} – ₹{ps2:,.0f}.
Consider students profile, strengths and weaknesses and Suggest **5 technical skills** and **5 soft skills** the user should improve to enhance their chances of reaching their target salary.
Be motivational but practical. Avoid generic advice and be concise.
While mentioning technical skills, mention the software and technology if required
"""
        else:
            prompt = f"""{user_context}
The user's target salary ₹{target_salary:,.2f} is **within or below** the expected range of ₹{ps1:,.0f} – ₹{ps2:,.0f}.
Consider students profile, strengths and weaknesses and Still, suggest **5 technical skills** and **5 soft skills** they can work on to maximize career growth. Consider students profile and suggets based on that
Be motivational but practical. Avoid generic advice and be concise
While mentioning technical skills, mention the software and technology if required
"""
        llm = ChatTogether(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.7,
            top_p=0.9,
            together_api_key="your-api-key-here"
        )
        response = llm.invoke(prompt).content

        st.success(f"\U0001F4B0 Estimated Salary Range: ₹{ps1:,.0f} – ₹{ps2:,.0f}")
        st.markdown("---")
        st.subheader("\U0001F9E0 Personalized Skill Suggestions")
        st.write(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
