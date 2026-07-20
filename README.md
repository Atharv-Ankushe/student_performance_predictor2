# 🎓 Student Performance Predictor

An interactive, AI-powered web application built with **Streamlit** and **scikit-learn** that predicts student academic outcomes using a trained Support Vector Classifier (SVC) model.

🔗 **Live Demo:** [studentperformancepredictor2.streamlit.app](https://studentperformancepredictor2.streamlit.app/)

---

## 📌 Features

* **Real-time Academic Prediction**: Evaluates whether a student is likely to **Pass** or is **At Risk of Failing**.
* **Interactive Controls**: Sidebar inputs allowing dynamic adjustments to student demographics, study habits, and academic performance.
* **Modern Dashboard UI**: Styled with custom CSS, dynamic metric cards, and clean visual indicators.
* **Confidence Scoring**: Displays the raw SVM decision function score to evaluate prediction confidence.

---

## 🛠️ Model & Feature Set

The prediction engine utilizes a **Support Vector Machine (SVM)** model trained on 9 key student attributes:

1. **Gender** (`gender`): `0` = Female, `1` = Male
2. **Age** (`age`): Student's age (10–30)
3. **Weekly Study Hours** (`study_hours_per_week`): Hours spent studying per week
4. **Attendance Rate** (`attendance_rate`): School attendance percentage (0–100%)
5. **Parent Education Level** (`parent_education`): High School, Associate, Bachelor, or Master/PhD
6. **Internet Access** (`internet_access`): `0` = No, `1` = Yes
7. **Extracurricular Activities** (`extracurricular`): `0` = No, `1` = Yes
8. **Previous Exam Score** (`previous_score`): Prior assessment score (0–100)
9. **Final Assessment Score** (`final_score`): Latest evaluation score (0–100)

---

## 📂 Repository Structure

```text
├── app.py                # Main Streamlit web application
├── svm_model.pkl          # Trained Support Vector Classifier model
├── requirements.txt      # Python dependencies for deployment
└── README.md             # Project documentation
