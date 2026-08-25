import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

GRADE_STYLES = {
    "A": {"bg": "#ecfdf5", "border": "#34d399", "color": "#047857", "glow": "rgba(52, 211, 153, 0.25)"},
    "B": {"bg": "#f0fdf4", "border": "#4ade80", "color": "#15803d", "glow": "rgba(74, 222, 128, 0.2)"},
    "C": {"bg": "#fffbeb", "border": "#fbbf24", "color": "#b45309", "glow": "rgba(251, 191, 36, 0.22)"},
    "D": {"bg": "#fff7ed", "border": "#fb923c", "color": "#c2410c", "glow": "rgba(251, 146, 60, 0.22)"},
    "F": {"bg": "#fef2f2", "border": "#f87171", "color": "#b91c1c", "glow": "rgba(248, 113, 113, 0.22)"},
}

OUTCOME_STYLES = {
    "success": {"bg": "#ecfdf5", "border": "#6ee7b7", "color": "#065f46"},
    "warning": {"bg": "#fffbeb", "border": "#fcd34d", "color": "#92400e"},
    "error": {"bg": "#fef2f2", "border": "#fca5a5", "color": "#991b1b"},
}

# --- Styling ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.08), transparent 28%),
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.06), transparent 24%),
            #f4f7fb;
    }

    .block-container {
        max-width: 980px;
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
    }

    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 45%, #155e75 100%);
        color: #ffffff;
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 45px rgba(13, 148, 136, 0.22);
    }

    .hero::before {
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
    }

    .hero::after {
        content: "";
        position: absolute;
        bottom: -60px;
        left: 20%;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.05);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #ecfeff;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 0.85rem;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        color: #ffffff !important;
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero p {
        position: relative;
        z-index: 1;
        color: #ccfbf1;
        font-size: 1.02rem;
        margin: 0;
        line-height: 1.7;
        max-width: 760px;
    }

    .hero-stats {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.15rem;
    }

    .hero-stat {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        padding: 0.4rem 0.85rem;
        font-size: 0.82rem;
        color: #f0fdfa;
        font-weight: 600;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e7edf5;
        border-radius: 18px;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }

    .panel-title {
        margin: 0 0 0.55rem 0;
        color: #0f172a;
        font-size: 1.08rem;
        font-weight: 800;
    }

    .panel-text {
        color: #64748b;
        line-height: 1.7;
        margin: 0 0 1rem 0;
        font-size: 0.95rem;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.8rem;
    }

    .info-pill {
        background: linear-gradient(180deg, #fafcff 0%, #f8fafc 100%);
        border: 1px solid #e8eef5;
        border-radius: 14px;
        padding: 0.95rem;
        min-height: 124px;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .info-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.06);
    }

    .info-pill .icon-wrap {
        width: 2.2rem;
        height: 2.2rem;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.05rem;
        margin-bottom: 0.55rem;
        background: #ecfeff;
    }

    .info-pill .title {
        font-weight: 700;
        color: #0f766e;
        font-size: 0.86rem;
        margin-bottom: 0.3rem;
    }

    .info-pill .desc {
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.45;
        margin: 0;
    }

    .how-it-works {
        background: linear-gradient(135deg, #ecfeff 0%, #f0fdfa 100%);
        border: 1px solid #99f6e4;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.1rem;
    }

    .how-it-works p {
        color: #334155;
        margin: 0;
        line-height: 1.7;
        font-size: 0.94rem;
    }

    [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e7edf5;
        border-radius: 20px;
        padding: 1.35rem 1.45rem 1.1rem 1.45rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
    }

    .form-section-label {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        color: #0f766e;
        font-weight: 800;
        font-size: 0.98rem;
        margin: 0.15rem 0 0.85rem 0;
    }

    .form-section-label .dot {
        width: 0.55rem;
        height: 0.55rem;
        border-radius: 50%;
        background: #14b8a6;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        font-weight: 600 !important;
        color: #334155 !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 0.78rem 1rem !important;
        background: linear-gradient(135deg, #14b8a6, #0f766e) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 10px 24px rgba(20, 184, 166, 0.28) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(20, 184, 166, 0.34) !important;
        color: white !important;
        border: none !important;
    }

    .results-panel {
        background: #ffffff;
        border: 1px solid #e7edf5;
        border-radius: 22px;
        padding: 1.4rem 1.45rem;
        margin-top: 0.5rem;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
    }

    .result-section-title {
        color: #0f172a;
        font-size: 1.05rem;
        font-weight: 800;
        margin: 0 0 0.9rem 0;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    .result-section-title .line {
        flex: 1;
        height: 1px;
        background: #e2e8f0;
    }

    .grade-box {
        text-align: center;
        padding: 2rem 1.5rem 1.6rem 1.5rem;
        border-radius: 20px;
        margin: 0 0 1rem 0;
        position: relative;
        overflow: hidden;
    }

    .grade-box::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top, rgba(255,255,255,0.65), transparent 55%);
        pointer-events: none;
    }

    .grade-box .label {
        position: relative;
        margin: 0;
        color: #64748b;
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .grade-box .grade {
        position: relative;
        font-size: 4.2rem;
        font-weight: 800;
        margin: 0.2rem 0 0 0;
        line-height: 1;
        letter-spacing: -0.03em;
    }

    .confidence-wrap {
        background: #f8fbff;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        margin-bottom: 1rem;
    }

    .confidence-wrap .label {
        color: #1e40af;
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 0.45rem;
    }

    .confidence-wrap .value {
        color: #1d4ed8;
        font-weight: 800;
        font-size: 1.05rem;
        margin-top: 0.35rem;
    }

    .outcome-banner {
        border-radius: 14px;
        padding: 0.95rem 1rem;
        margin-bottom: 1.1rem;
        font-size: 0.94rem;
        line-height: 1.55;
        font-weight: 600;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e7edf5;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
        min-height: 92px;
    }

    .metric-card.math { border-top: 4px solid #3b82f6; }
    .metric-card.science { border-top: 4px solid #10b981; }
    .metric-card.english { border-top: 4px solid #8b5cf6; }
    .metric-card.average { border-top: 4px solid #0f766e; }

    .metric-card .label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }

    .metric-card .value {
        color: #0f172a;
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1;
    }

    .profile-note {
        color: #64748b;
        font-size: 0.86rem;
        font-style: italic;
        margin: -0.25rem 0 0.85rem 0;
    }

    .profile-card {
        border-radius: 16px;
        padding: 1.05rem 1.15rem;
        min-height: 190px;
        border: 1px solid transparent;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }

    .profile-card.strengths {
        background: linear-gradient(180deg, #ecfdf5 0%, #f8fffb 100%);
        border-color: #a7f3d0;
    }

    .profile-card.support {
        background: linear-gradient(180deg, #fff7ed 0%, #fffaf5 100%);
        border-color: #fdba74;
    }

    .profile-card h4 {
        margin: 0 0 0.7rem 0;
        font-size: 1rem;
        font-weight: 800;
    }

    .profile-card.strengths h4 { color: #047857; }
    .profile-card.support h4 { color: #c2410c; }

    .profile-card ul {
        margin: 0;
        padding-left: 1.1rem;
        color: #334155;
        line-height: 1.7;
        font-size: 0.92rem;
    }

    .download-wrap {
        margin-top: 0.35rem;
        padding: 1rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        text-align: center;
    }

    .download-wrap p {
        margin: 0 0 0.65rem 0;
        color: #64748b;
        font-size: 0.88rem;
    }

    div[data-testid="stDownloadButton"] > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        background: #ffffff !important;
        color: #0f766e !important;
        border: 1px solid #99f6e4 !important;
        box-shadow: none !important;
    }

    div[data-testid="stDownloadButton"] > button:hover {
        background: #f0fdfa !important;
        color: #0f766e !important;
        border: 1px solid #5eead4 !important;
        transform: none !important;
    }

    .footer-note {
        color: #94a3b8;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 2rem;
        line-height: 1.65;
        padding: 0 1rem;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-weight: 700;
        border-radius: 10px 10px 0 0;
    }

    @media (max-width: 900px) {
        .info-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 600px) {
        .hero {
            padding: 1.5rem 1.25rem;
        }

        .hero h1 {
            font-size: 1.7rem;
        }

        .info-grid {
            grid-template-columns: 1fr;
        }

        .grade-box .grade {
            font-size: 3.4rem;
        }
    }
</style>
""", unsafe_allow_html=True)


def render_outcome_banner(message, tone):
    style = OUTCOME_STYLES[tone]
    st.markdown(f"""
    <div class="outcome-banner" style="
        background:{style['bg']};
        border:1px solid {style['border']};
        color:{style['color']};
    ">{message}</div>
    """, unsafe_allow_html=True)


def render_metric_card(label, value, css_class):
    st.markdown(f"""
    <div class="metric-card {css_class}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


SELECT_PLACEHOLDER = "— Select —"


def validate_form_fields(fields):
    missing = [label for label, value in fields if value is None or value == SELECT_PLACEHOLDER]
    return missing


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("student_performance_preprocessor.joblib")
    model = joblib.load("student_performance_best_model.joblib")
    return preprocessor, model


try:
    preprocessor, model = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files were not found. Upload "
        "`student_performance_preprocessor.joblib` and "
        "`student_performance_best_model.joblib` to the same folder as `app.py`."
    )
    st.stop()


# --- Header ---
st.markdown("""
<div class="hero">
    <div class="hero-badge">ML-Powered Educational Tool</div>
    <h1>🎓 Student Grade Predictor</h1>
    <p>
        Estimate a student's likely final grade from study habits, attendance, and subject
        scores — using patterns learned from historical student records.
    </p>
    <div class="hero-stats">
        <span class="hero-stat">12 model features</span>
        <span class="hero-stat">A–F grade output</span>
        <span class="hero-stat">Student-support insights</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_about = st.tabs(["🔮 Predict Grade", "📖 About the Data"])

with tab_about:
    st.markdown("""
    <div class="panel">
        <p class="panel-title">📊 What data powers this prediction?</p>
        <p class="panel-text">
            This tool is built from individually structured student records, where each row
            represents one student with their demographic profile, educational background,
            learning habits, and academic performance. The dataset blends behavioral,
            environmental, and academic factors — making it useful for educational analysis
            and student-support planning.
        </p>
        <div class="info-grid">
            <div class="info-pill">
                <div class="icon-wrap">👤</div>
                <div class="title">Demographics</div>
                <p class="desc">Age, gender, and school type</p>
            </div>
            <div class="info-pill">
                <div class="icon-wrap">🏠</div>
                <div class="title">Family Background</div>
                <p class="desc">Parent education level</p>
            </div>
            <div class="info-pill">
                <div class="icon-wrap">📖</div>
                <div class="title">Study Habits</div>
                <p class="desc">Daily study hours, study method, internet access</p>
            </div>
            <div class="info-pill">
                <div class="icon-wrap">🏫</div>
                <div class="title">School Engagement</div>
                <p class="desc">Attendance, travel time, extra activities</p>
            </div>
            <div class="info-pill">
                <div class="icon-wrap">📝</div>
                <div class="title">Academic Records</div>
                <p class="desc">Marks in Math, Science, and English</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-it-works">
        <p>
            <strong>What does this prediction do?</strong> The trained machine learning model
            estimates a student's <strong>final letter grade</strong> (A–F) from 12 selected
            inputs. It does <em>not</em> replace official grading — it highlights patterns
            from past students so teachers and advisors can identify who may benefit from
            extra support early.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ About this prediction system"):
        st.markdown("""
This system predicts a student's **final grade** using **12 selected features** from the
student profile and academic records entered in the form.

Each training record represents one student with demographic, behavioral, environmental,
and academic information — similar to the categories shown above.

**Features used:** age, gender, school type, parent education level, daily study hours,
study method, internet access, attendance percentage, extra activities, and scores in
math, science, and English.

**Excluded from the model:** student identifiers, travel time, and overall score.
Overall score is intentionally excluded to **reduce target leakage**, since it is closely
related to the final grade being predicted.

**Model selection:** Logistic Regression, K-Nearest Neighbors (KNN), Support Vector
Machine (SVM), and Artificial Neural Network (ANN) were trained and compared. The best
performing model was selected using **weighted F1-score**.
        """)

with tab_predict:
    st.markdown("""
    <div class="how-it-works">
        <p>
            Fill in the student profile below, then click <strong>Predict Final Grade</strong>
            to generate a grade estimate, confidence score, academic summary, and learning profile.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("student_prediction_form"):
        st.markdown(
            '<p class="form-section-label"><span class="dot"></span> Student Profile</p>',
            unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age", min_value=1, max_value=100, value=None, placeholder="Enter age", step=1
            )
            gender = st.selectbox(
                "Gender", [SELECT_PLACEHOLDER, "Female", "Male", "Other"]
            )
            school_type = st.selectbox(
                "School Type", [SELECT_PLACEHOLDER, "Public", "Private"]
            )

        with col2:
            parent_education = st.selectbox(
                "Parent Education Level",
                [SELECT_PLACEHOLDER, "High School", "Diploma", "Bachelor", "Master", "PhD"]
            )
            internet_access = st.selectbox(
                "Internet Access", [SELECT_PLACEHOLDER, "Yes", "No"]
            )
            extra_activities = st.selectbox(
                "Extra Activities", [SELECT_PLACEHOLDER, "Yes", "No"]
            )

        st.divider()
        st.markdown(
            '<p class="form-section-label"><span class="dot"></span> Study & Academic Records</p>',
            unsafe_allow_html=True
        )
        col3, col4 = st.columns(2)

        with col3:
            study_hours = st.number_input(
                "Daily Study Hours",
                min_value=0.0,
                max_value=24.0,
                value=None,
                placeholder="Enter hours",
                step=0.5
            )
            study_method = st.selectbox(
                "Study Method",
                [SELECT_PLACEHOLDER, "Self-study", "Group Study", "Online Learning", "Tutoring"]
            )
            attendance_percentage = st.slider(
            "Attendance Percentage",
                min_value=0,
                max_value=100,
                value=75,
                step=1,
                format="%d%%"
            )

        with col4:
            math_score = st.number_input(
                "Math Score", 0.0, 100.0, value=None, placeholder="Enter score", step=1.0
            )
            science_score = st.number_input(
                "Science Score", 0.0, 100.0, value=None, placeholder="Enter score", step=1.0
            )
            english_score = st.number_input(
                "English Score", 0.0, 100.0, value=None, placeholder="Enter score", step=1.0
            )

        st.write("")
        submitted = st.form_submit_button("✨ Predict Final Grade")

    if submitted:
        missing_fields = validate_form_fields([
            ("Age", age),
            ("Gender", gender),
            ("School Type", school_type),
            ("Parent Education Level", parent_education),
            ("Internet Access", internet_access),
            ("Extra Activities", extra_activities),
            ("Daily Study Hours", study_hours),
            ("Study Method", study_method),
            ("Attendance Percentage", attendance_percentage),
            ("Math Score", math_score),
            ("Science Score", science_score),
            ("English Score", english_score),
        ])

        if missing_fields:
            st.warning(f"Please complete all fields before predicting: {', '.join(missing_fields)}")
        else:
            new_student = pd.DataFrame([{
                "age": age,
                "gender": gender,
                "school_type": school_type,
                "parent_education": parent_education,
                "study_hours": study_hours,
                "study_method": study_method,
                "internet_access": internet_access,
                "attendance_percentage": attendance_percentage,
                "extra_activities": extra_activities,
                "math_score": math_score,
                "science_score": science_score,
                "english_score": english_score
            }])

            processed_student = preprocessor.transform(new_student)
            prediction = model.predict(processed_student)[0]
            grade = str(prediction).strip().upper()
            average_subject_score = (math_score + science_score + english_score) / 3
            grade_style = GRADE_STYLES.get(grade, GRADE_STYLES["C"])

            st.markdown('<div class="results-panel">', unsafe_allow_html=True)
            st.markdown(
                '<p class="result-section-title">🎯 Prediction Result <span class="line"></span></p>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="grade-box" style="
                background: linear-gradient(180deg, {grade_style['bg']} 0%, #ffffff 100%);
                border: 1px solid {grade_style['border']};
                box-shadow: 0 12px 30px {grade_style['glow']};
            ">
                <p class="label">Predicted Final Grade</p>
                <p class="grade" style="color:{grade_style['color']};">{grade}</p>
            </div>
            """, unsafe_allow_html=True)

            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(processed_student)[0]
                    class_labels = model.classes_
                    pred_idx = list(class_labels).index(prediction)
                    confidence = proba[pred_idx]
                    st.markdown(f"""
                    <div class="confidence-wrap">
                        <div class="label">Model Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(confidence))
                    st.markdown(
                        f'<p class="value" style="color:#1d4ed8; font-weight:800; margin-top:-0.35rem;">'
                        f"{confidence * 100:.1f}%</p>",
                        unsafe_allow_html=True
                    )
                except (AttributeError, IndexError, ValueError, TypeError):
                    pass

            if grade in ["A", "B"]:
                render_outcome_banner(
                    "Strong predicted outcome — patterns associated with higher performance.",
                    "success"
                )
            elif grade == "C":
                render_outcome_banner(
                    "Moderate predicted outcome — consistent study support may help.",
                    "warning"
                )
            else:
                render_outcome_banner(
                    "May benefit from additional academic support or attendance monitoring.",
                    "error"
                )

            st.markdown(
                '<p class="result-section-title">📈 Academic Summary <span class="line"></span></p>',
                unsafe_allow_html=True
            )
            sum1, sum2, sum3, sum4 = st.columns(4)
            with sum1:
                render_metric_card("Math Score", f"{math_score:.1f}", "math")
            with sum2:
                render_metric_card("Science Score", f"{science_score:.1f}", "science")
            with sum3:
                render_metric_card("English Score", f"{english_score:.1f}", "english")
            with sum4:
                render_metric_card("Average Subject Score", f"{average_subject_score:.1f}", "average")

            st.markdown(
                '<p class="result-section-title">🧭 Learning Profile <span class="line"></span></p>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<p class="profile-note">These are input-based indicators, not direct '
                "explanations of the model prediction.</p>",
                unsafe_allow_html=True
            )

            strengths = []
            areas_for_support = []

            if average_subject_score >= 75:
                strengths.append("Strong overall subject performance")
            if average_subject_score < 60:
                areas_for_support.append("Subject-score improvement may be needed")
            if attendance_percentage >= 90:
                strengths.append("Strong attendance record")
            if attendance_percentage < 75:
                areas_for_support.append("Attendance may need improvement")
            if study_hours >= 2:
                strengths.append("Consistent daily study time")
            if study_hours < 2:
                areas_for_support.append("More regular study time may be beneficial")
            if internet_access == "Yes":
                strengths.append("Access to online learning resources")

            profile_left, profile_right = st.columns(2)

            strengths_html = "".join(f"<li>{item}</li>" for item in strengths) or (
                "<li><em>No specific strengths identified from inputs.</em></li>"
            )
            support_html = "".join(f"<li>{item}</li>" for item in areas_for_support) or (
                "<li><em>No specific support areas identified from inputs.</em></li>"
            )

            with profile_left:
                st.markdown(f"""
                <div class="profile-card strengths">
                    <h4>✅ Strengths</h4>
                    <ul>{strengths_html}</ul>
                </div>
                """, unsafe_allow_html=True)

            with profile_right:
                st.markdown(f"""
                <div class="profile-card support">
                    <h4>🛟 Areas for Support</h4>
                    <ul>{support_html}</ul>
                </div>
                """, unsafe_allow_html=True)

            report_df = pd.DataFrame([{
                "predicted_final_grade": grade,
                "age": age,
                "gender": gender,
                "school_type": school_type,
                "parent_education": parent_education,
                "study_hours": study_hours,
                "study_method": study_method,
                "internet_access": internet_access,
                "attendance_percentage": attendance_percentage,
                "extra_activities": extra_activities,
                "math_score": math_score,
                "science_score": science_score,
                "english_score": english_score,
                "average_subject_score": average_subject_score
            }])

            st.markdown(
                '<p class="result-section-title">📄 Export Report <span class="line"></span></p>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="download-wrap"><p>Save this prediction and all entered student inputs as a CSV file.</p></div>',
                unsafe_allow_html=True
            )
            st.download_button(
                label="⬇️ Download Prediction Report",
                data=report_df.to_csv(index=False),
                file_name="student_grade_prediction.csv",
                mime="text/csv"
            )

            st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<p class="footer-note">Disclaimer: This prediction is generated from patterns in the '
    "training dataset. It is intended for educational analysis and student-support "
    "purposes only, not as a final academic decision.</p>",
    unsafe_allow_html=True
)
