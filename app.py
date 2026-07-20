import pickle
import numpy as np
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM BEAUTIFUL CSS STYLING ---
st.markdown(
    """
<style>
    /* Main container background */
    .main {
        background-color: #0E1117;
    }
    
    /* Header Styling */
    .title-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .subtitle-header {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 35px;
    }

    /* Prediction Card */
    .result-card-pass {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
        border: 1px solid #10B981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }
    
    .result-card-fail {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
        border: 1px solid #EF4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .pass-text { color: #10B981; }
    .fail-text { color: #EF4444; }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #6366F1, #3B82F6);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #4F46E5, #2563EB);
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load model from the pickle file in your repo
@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

# Header Section
st.markdown(
    '<div class="title-header">Student Success Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle-header">Predict academic outcome using your trained Support Vector Classifier</div>',
    unsafe_allow_html=True,
)

# App Sidebar / Controls
with st.sidebar:
    st.header("📋 Input Parameters")
    st.markdown("Adjust student data below:")

    gender = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
    )
    age = st.slider("Age", min_value=10, max_value=30, value=17)
    study_hours = st.slider(
        "Study Hours / Week", min_value=0, max_value=60, value=20
    )
    attendance_rate = st.slider(
        "Attendance Rate (%)", min_value=0, max_value=100, value=85
    )
    parent_education = st.selectbox(
        "Parent Education Level",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "High School",
            "Associate Degree",
            "Bachelor Degree",
            "Master/PhD",
        ][x],
    )
    internet_access = st.radio(
        "Internet Access",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
    )
    extracurricular = st.radio(
        "Extracurricular Activities",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
    )
    previous_score = st.number_input(
        "Previous Exam Score", min_value=0.0, max_value=100.0, value=75.0
    )
    final_score = st.number_input(
        "Final Assessment Score", min_value=0.0, max_value=100.0, value=78.0
    )

# Main Dashboard View
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📊 Student Overview Summary")

    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Attendance Rate", f"{attendance_rate}%")
    m_col2.metric("Weekly Study Hours", f"{study_hours} hrs")

    m_col3, m_col4 = st.columns(2)
    m_col3.metric("Previous Score", f"{previous_score:.1f}")
    m_col4.metric("Final Score", f"{final_score:.1f}")

    predict_btn = st.button("🚀 Run Assessment Model")

with col2:
    st.subheader("🎯 Prediction Output")

    if predict_btn:
        input_data = np.array(
            [
                [
                    gender,
                    age,
                    study_hours,
                    attendance_rate,
                    parent_education,
                    internet_access,
                    extracurricular,
                    previous_score,
                    final_score,
                ]
            ]
        )

        prediction = model.predict(input_data)[0]
        decision_score = model.decision_function(input_data)[0]

        if prediction == "Yes":
            st.markdown(
                f"""
            <div class="result-card-pass">
                <div class="result-title pass-text">🎉 PASS CONFIRMED</div>
                <p style="color:#D1D5DB; font-size:1.05rem;">
                    The model predicts that this student will <b>PASS</b>.
                </p>
                <p style="color:#9CA3AF; font-size:0.9rem;">
                    Decision Function Confidence Score: <b>{decision_score:.3f}</b>
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="result-card-fail">
                <div class="result-title fail-text">⚠️ AT-RISK DETECTED</div>
                <p style="color:#D1D5DB; font-size:1.05rem;">
                    The model predicts that this student is <b>AT RISK OF FAILING</b>.
                </p>
                <p style="color:#9CA3AF; font-size:0.9rem;">
                    Decision Function Confidence Score: <b>{decision_score:.3f}</b>
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.info("👈 Adjust parameters on the sidebar and click **Run Assessment Model**.")
