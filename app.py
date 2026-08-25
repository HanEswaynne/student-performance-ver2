import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OUTCOME_STYLES = {
    "pass": {
        "bg": "#ecfdf5",
        "border": "#34d399",
        "color": "#047857",
        "glow": "rgba(52, 211, 153, 0.25)",
    },
    "support": {
        "bg": "#fef2f2",
        "border": "#f87171",
        "color": "#b91c1c",
        "glow": "rgba(248, 113, 113, 0.22)",
    },
}

SELECT_PLACEHOLDER = "— Select —"

st.markdown(
    """
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
    background: rgba(255,255,255,.08);
}

.hero::after {
    content: "";
    position: absolute;
    bottom: -60px;
    left: 20%;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,.05);
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.22);
    color: #ecfeff;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
    padding: .35rem .75rem;
    border-radius: 999px;
    margin-bottom: .85rem;
    position: relative;
    z-index: 1;
}

.hero h1 {
    position: relative;
    z-index: 1;
    color: #fff !important;
    font-size: 2.15rem;
    font-weight: 800;
    margin-bottom: .5rem;
    letter-spacing: -.02em;
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
    gap: .65rem;
    margin-top: 1.15rem;
}

.hero-stat {
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 999px;
    padding: .4rem .85rem;
    font-size: .82rem;
    color: #f0fdfa;
    font-weight: 600;
}

.panel,
[data-testid="stForm"],
.results-panel {
    background: #fff;
    border: 1px solid #e7edf5;
    border-radius: 20px;
    padding: 1.35rem 1.45rem;
    box-shadow: 0 10px 28px rgba(15,23,42,.05);
}

.panel {
    margin-bottom: 1rem;
}

.panel-title {
    margin: 0 0 .55rem;
    color: #0f172a;
    font-size: 1.08rem;
    font-weight: 800;
}

.panel-text {
    color: #64748b;
    line-height: 1.7;
    margin: 0 0 1rem;
    font-size: .95rem;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: .8rem;
}

.info-pill {
    background: linear-gradient(180deg,#fafcff 0%,#f8fafc 100%);
    border: 1px solid #e8eef5;
    border-radius: 14px;
    padding: .95rem;
    min-height: 124px;
}

.icon-wrap {
    width: 2.2rem;
    height: 2.2rem;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    margin-bottom: .55rem;
    background: #ecfeff;
}

.info-pill .title {
    font-weight: 700;
    color: #0f766e;
    font-size: .86rem;
    margin-bottom: .3rem;
}

.info-pill .desc {
    color: #64748b;
    font-size: .78rem;
    line-height: 1.45;
    margin: 0;
}

.how-it-works {
    background: linear-gradient(135deg,#ecfeff 0%,#f0fdfa 100%);
    border: 1px solid #99f6e4;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.1rem;
}

.how-it-works p {
    color: #334155;
    margin: 0;
    line-height: 1.7;
    font-size: .94rem;
}

.form-section-label {
    display: flex;
    align-items: center;
    gap: .45rem;
    color: #0f766e;
    font-weight: 800;
    font-size: .98rem;
    margin: .15rem 0 .85rem;
}

.dot {
    width: .55rem;
    height: .55rem;
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
    padding: .78rem 1rem !important;
    background: linear-gradient(135deg,#14b8a6,#0f766e) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 10px 24px rgba(20,184,166,.28) !important;
}

.results-panel {
    margin-top: .5rem;
    border-radius: 22px;
}

.result-section-title {
    color: #0f172a;
    font-size: 1.05rem;
    font-weight: 800;
    margin: 0 0 .9rem;
    display: flex;
    align-items: center;
    gap: .45rem;
}

.line {
    flex: 1;
    height: 1px;
    background: #e2e8f0;
}

.outcome-box {
    text-align: center;
    padding: 2rem 1.5rem 1.6rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

.outcome-label {
    margin: 0;
    color: #64748b;
    font-size: .82rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    font-weight: 700;
}

.outcome-value {
    font-size: 2.4rem;
    font-weight: 800;
    margin: .35rem 0 0;
    line-height: 1.1;
}

.confidence-wrap {
    background: #f8fbff;
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: .95rem 1rem;
    margin-bottom: 1rem;
}

.confidence-wrap .label {
    color: #1e40af;
    font-weight: 700;
    font-size: .88rem;
    margin-bottom: .45rem;
}

.outcome-banner {
    border-radius: 14px;
    padding: .95rem 1rem;
    margin-bottom: 1.1rem;
    font-size: .94rem;
    line-height: 1.55;
    font-weight: 600;
}

.metric-card {
    background: #fff;
    border: 1px solid #e7edf5;
    border-radius: 14px;
    padding: .95rem 1rem;
    box-shadow: 0 4px 14px rgba(15,23,42,.03);
    min-height: 92px;
}

.metric-card.attendance {
    border-top: 4px solid #3b82f6;
}

.metric-card.study {
    border-top: 4px solid #10b981;
}

.metric-card.grade {
    border-top: 4px solid #8b5cf6;
}

.metric-card.activities {
    border-top: 4px solid #0f766e;
}

.metric-card .label {
    color: #64748b;
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .04em;
    margin-bottom: .35rem;
}

.metric-card .value {
    color: #0f172a;
    font-size: 1.45rem;
    font-weight: 800;
    line-height: 1.1;
}

.profile-card {
    border-radius: 16px;
    padding: 1.05rem 1.15rem;
    min-height: 180px;
    border: 1px solid transparent;
    box-shadow: 0 6px 18px rgba(15,23,42,.04);
}

.profile-card.strengths {
    background: linear-gradient(180deg,#ecfdf5 0%,#f8fffb 100%);
    border-color: #a7f3d0;
}

.profile-card.support {
    background: linear-gradient(180deg,#fff7ed 0%,#fffaf5 100%);
    border-color: #fdba74;
}

.profile-card h4 {
    margin: 0 0 .7rem;
    font-size: 1rem;
    font-weight: 800;
}

.profile-card.strengths h4 {
    color: #047857;
}

.profile-card.support h4 {
    color: #c2410c;
}

.profile-card ul {
    margin: 0;
    padding-left: 1.1rem;
    color: #334155;
    line-height: 1.7;
    font-size: .92rem;
}

.download-wrap {
    margin-top: .35rem;
    padding: 1rem;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    text-align: center;
}

.download-wrap p {
    margin: 0 0 .65rem;
    color: #64748b;
    font-size: .88rem;
}

div[data-testid="stDownloadButton"] > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    background: #fff !important;
    color: #0f766e !important;
    border: 1px solid #99f6e4 !important;
    box-shadow: none !important;
}

.footer-note {
    color: #94a3b8;
    font-size: .82rem;
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

    .outcome-value {
        font-size: 2rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("student_passfail_preprocessor.joblib")
    model = joblib.load("student_passfail_best_model.joblib")
    selected_features = joblib.load("student_passfail_features.joblib")

    return preprocessor, model, selected_features


def render_metric(label, value, css_class):
    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_banner(message, tone):
    if tone == "pass":
        bg, border, color = "#ecfdf5", "#6ee7b7", "#065f46"
    else:
        bg, border, color = "#fef2f2", "#fca5a5", "#991b1b"

    st.markdown(
        f"""
        <div class="outcome-banner"
             style="background:{bg}; border:1px solid {border}; color:{color};">
             {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def missing_fields(fields):
    return [
        label
        for label, value in fields
        if value is None or value == SELECT_PLACEHOLDER
    ]


try:
    preprocessor, model, selected_features = load_artifacts()

except FileNotFoundError:
    st.error(
        "Model files were not found. Put these files in the same folder as `app.py`: "
        "`student_passfail_preprocessor.joblib`, "
        "`student_passfail_best_model.joblib`, and "
        "`student_passfail_features.joblib`."
    )
    st.stop()


st.markdown(
    f"""
<div class="hero">
    <div class="hero-badge">ML-Powered Educational Tool</div>

    <h1>🎓 Student Pass/Fail Predictor</h1>

    <p>
        Estimate whether a student is likely to pass or may need academic support,
        based on attendance, weekly study time, previous grade,
        extracurricular activities, parental support, and gender.
    </p>

    <div class="hero-stats">
        <span class="hero-stat">{len(selected_features)} model features</span>
        <span class="hero-stat">Pass / Needs Support output</span>
        <span class="hero-stat">Student-support insights</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_predict, tab_about = st.tabs(
    ["🔮 Predict Outcome", "📖 About the Data"]
)

with tab_about:
    st.markdown(
        """
<div class="panel">
    <p class="panel-title">📊 What data powers this prediction?</p>

    <p class="panel-text">
        Each record represents one student. The prediction model was trained using
        student engagement, learning habits, prior academic performance,
        parental support, extracurricular participation, and gender.
        Student names and IDs were excluded because they are identifiers rather
        than meaningful predictors.
    </p>

    <div class="info-grid">
        <div class="info-pill">
            <div class="icon-wrap">👤</div>
            <div class="title">Student Profile</div>
            <p class="desc">Gender as recorded in the dataset</p>
        </div>

        <div class="info-pill">
            <div class="icon-wrap">🏫</div>
            <div class="title">Attendance</div>
            <p class="desc">Attendance rate as a percentage</p>
        </div>

        <div class="info-pill">
            <div class="icon-wrap">📖</div>
            <div class="title">Study Habits</div>
            <p class="desc">Study hours completed each week</p>
        </div>

        <div class="info-pill">
            <div class="icon-wrap">📝</div>
            <div class="title">Previous Grade</div>
            <p class="desc">Prior academic grade or score</p>
        </div>

        <div class="info-pill">
            <div class="icon-wrap">🤝</div>
            <div class="title">Support & Activities</div>
            <p class="desc">Parental support and extracurricular activities</p>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="how-it-works">
    <p>
        <strong>What does this prediction do?</strong>
        The model estimates whether a student is likely to <strong>pass</strong>
        or may <strong>need academic support</strong>. It is an educational-support
        indicator based on patterns in the training data, not an official result
        or final academic decision.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ About this prediction system"):
        st.markdown(
            f"""
This application uses the selected features saved during training:

```python
{selected_features}
```

The notebook compared:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Artificial Neural Network

The deployed model is the algorithm that achieved the highest F1-score
on the test dataset.

Target labels:

- **Pass (1)**
- **Needs Academic Support / Fail (0)**
"""
        )

with tab_predict:
    st.markdown(
        """
<div class="how-it-works">
    <p>
        Enter the student information below, then click
        <strong>Predict Student Outcome</strong> to view the predicted outcome,
        model confidence, input summary, and learning-support indicators.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("student_passfail_form"):
        st.markdown(
            '<p class="form-section-label"><span class="dot"></span> Student Profile</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                [SELECT_PLACEHOLDER, "Female", "Male"],
            )

            parental_support = st.selectbox(
                "Parental Support",
                [SELECT_PLACEHOLDER, "Low", "Medium", "High"],
            )

        with col2:
            extracurricular_activities = st.number_input(
                "Extracurricular Activities (Number)",
                min_value=0,
                max_value=3,
                value=None,
                placeholder="Enter number of activities",
                step=1,
            )

        st.divider()

        st.markdown(
            '<p class="form-section-label"><span class="dot"></span> Attendance & Academic Records</p>',
            unsafe_allow_html=True,
        )

        col3, col4 = st.columns(2)

        with col3:
            attendance_rate = st.slider(
                "Attendance Rate",
                min_value=0,
                max_value=100,
                value=75,
                step=1,
                format="%d%%",
            )

            study_hours = st.number_input(
                "Study Hours Per Week",
                min_value=0,
                max_value=100,
                value=None,
                placeholder="Enter weekly study hours",
                step=1,
            )

        with col4:
            previous_grade = st.number_input(
                "Previous Grade",
                min_value=0,
                max_value=100,
                value=None,
                placeholder="Enter previous grade",
                step=1,
            )

            st.info(
                "All fields are required before a prediction can be generated."
            )

        st.write("")

        submitted = st.form_submit_button(
            "✨ Predict Student Outcome"
        )

    if submitted:
        missing = missing_fields(
            [
                ("Gender", gender),
                ("Parental Support", parental_support),
                ("Extracurricular Activities", extracurricular_activities),
                ("Study Hours Per Week", study_hours),
                ("Previous Grade", previous_grade),
            ]
        )

        if missing:
            st.warning(
                "Please complete all fields before predicting: "
                + ", ".join(missing)
            )

        else:
            input_data = {
                "Gender": gender,
                "AttendanceRate": attendance_rate,
                "StudyHoursPerWeek": study_hours,
                "PreviousGrade": previous_grade,
                "ExtracurricularActivities": extracurricular_activities,
                "ParentalSupport": parental_support,
            }

            # Keep only the exact columns and column order
            # used by the final trained model.
            new_student = pd.DataFrame([input_data])
            new_student = new_student.reindex(
                columns=selected_features
            )

            processed_student = preprocessor.transform(
                new_student
            )

            prediction = int(
                model.predict(processed_student)[0]
            )

            is_pass = prediction == 1
            outcome = "PASS" if is_pass else "NEEDS SUPPORT"
            tone = "pass" if is_pass else "support"
            style = OUTCOME_STYLES[tone]

            confidence_percent = None

            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(
                        processed_student
                    )[0]

                    class_index = list(model.classes_).index(
                        prediction
                    )

                    confidence_percent = round(
                        float(probabilities[class_index]) * 100,
                        2,
                    )

                except (
                    AttributeError,
                    IndexError,
                    TypeError,
                    ValueError,
                ):
                    confidence_percent = None

            st.markdown(
                '<div class="results-panel">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<p class="result-section-title">🎯 Prediction Result '
                '<span class="line"></span></p>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
<div class="outcome-box"
     style="background:linear-gradient(180deg,{style['bg']} 0%,#ffffff 100%);
            border:1px solid {style['border']};
            box-shadow:0 12px 30px {style['glow']};">

    <p class="outcome-label">Predicted Student Outcome</p>

    <p class="outcome-value"
       style="color:{style['color']};">
       {outcome}
    </p>
</div>
""",
                unsafe_allow_html=True,
            )

            if confidence_percent is not None:
                st.markdown(
                    """
<div class="confidence-wrap">
    <div class="label">Model Confidence</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                st.progress(confidence_percent / 100)

                st.markdown(
                    f"""
<p style="color:#1d4ed8;font-weight:800;margin-top:-.35rem;">
    {confidence_percent:.1f}%
</p>
""",
                    unsafe_allow_html=True,
                )

            if is_pass:
                render_banner(
                    "Positive predicted outcome — the entered profile aligns "
                    "with patterns associated with passing in the training data.",
                    "pass",
                )
            else:
                render_banner(
                    "The model indicates that this student may benefit from "
                    "early academic support, study planning, or attendance monitoring.",
                    "support",
                )

            st.markdown(
                '<p class="result-section-title">📈 Student Input Summary '
                '<span class="line"></span></p>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                render_metric(
                    "Attendance Rate",
                    f"{attendance_rate}%",
                    "attendance",
                )

            with m2:
                render_metric(
                    "Study Hours / Week",
                    str(study_hours),
                    "study",
                )

            with m3:
                render_metric(
                    "Previous Grade",
                    str(previous_grade),
                    "grade",
                )

            with m4:
                render_metric(
                    "Activities",
                    str(extracurricular_activities),
                    "activities",
                )

            strengths = []
            support_areas = []

            if attendance_rate >= 90:
                strengths.append("Strong attendance record")
            elif attendance_rate < 75:
                support_areas.append("Attendance may need improvement")

            if study_hours >= 10:
                strengths.append("Consistent weekly study time")
            elif study_hours < 5:
                support_areas.append(
                    "More regular weekly study time may help"
                )

            if previous_grade >= 75:
                strengths.append(
                    "Strong previous academic performance"
                )
            elif previous_grade < 60:
                support_areas.append(
                    "Previous grade suggests a need for academic support"
                )

            if parental_support == "High":
                strengths.append(
                    "High parental support is available"
                )
            elif parental_support == "Low":
                support_areas.append(
                    "Consider strengthening learning support at home or school"
                )

            if extracurricular_activities > 0:
                strengths.append(
                    "Involvement in extracurricular activities"
                )

            if not strengths:
                strengths.append(
                    "No specific strengths were identified using the simple input indicators"
                )

            if not support_areas:
                support_areas.append(
                    "No immediate support area was identified using the simple input indicators"
                )

            strengths_html = "".join(
                f"<li>{item}</li>"
                for item in strengths
            )

            support_html = "".join(
                f"<li>{item}</li>"
                for item in support_areas
            )

            st.markdown(
                '<p class="result-section-title">🧭 Learning Profile '
                '<span class="line"></span></p>',
                unsafe_allow_html=True,
            )

            left, right = st.columns(2)

            with left:
                st.markdown(
                    f"""
<div class="profile-card strengths">
    <h4>✅ Strengths</h4>
    <ul>{strengths_html}</ul>
</div>
""",
                    unsafe_allow_html=True,
                )

            with right:
                st.markdown(
                    f"""
<div class="profile-card support">
    <h4>🛟 Areas for Support</h4>
    <ul>{support_html}</ul>
</div>
""",
                    unsafe_allow_html=True,
                )

            report = pd.DataFrame(
                [
                    {
                        "predicted_outcome": (
                            "Pass"
                            if is_pass
                            else "Needs Academic Support"
                        ),
                        "model_confidence_percent": confidence_percent,
                        "gender": gender,
                        "attendance_rate": attendance_rate,
                        "study_hours_per_week": study_hours,
                        "previous_grade": previous_grade,
                        "extracurricular_activities": extracurricular_activities,
                        "parental_support": parental_support,
                    }
                ]
            )

            st.markdown(
                '<p class="result-section-title">📄 Export Report '
                '<span class="line"></span></p>',
                unsafe_allow_html=True,
            )

            st.markdown(
                """
<div class="download-wrap">
    <p>Save this prediction and all entered student inputs as a CSV file.</p>
</div>
""",
                unsafe_allow_html=True,
            )

            st.download_button(
                label="⬇️ Download Prediction Report",
                data=report.to_csv(index=False),
                file_name="student_passfail_prediction.csv",
                mime="text/csv",
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

st.markdown(
    """
<p class="footer-note">
    Disclaimer: This prediction is generated from patterns in the training dataset.
    It is intended for educational analysis and student-support purposes only,
    not as a final academic decision.
</p>
""",
    unsafe_allow_html=True,
)
