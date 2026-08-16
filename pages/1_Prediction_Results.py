import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIGURATION
# ==========================================
CHART_ICON = "\U0001F4CA"   
st.set_page_config(
    page_title="Prediction Results",
    page_icon=CHART_ICON,
    layout="wide"
)

# ==========================================
# UNICODE ICONS
# ==========================================

SCHOOL_ICON = "\U0001F3EB"          
STUDENT_ICON = "\U0001F393"         
SUCCESS_ICON = "\u2705"             
TABLE_ICON = "\U0001F4CB"           
CHART_ICON = "\U0001F4CA"           
DOWNLOAD_ICON = "\U0001F4E5"        
BACK_ICON = "\u2B05\uFE0F"          
GREEN_ICON = "\U0001F7E2"           
YELLOW_ICON = "\U0001F7E1"          
RED_ICON = "\U0001F534"             
PERSON_ICON = "\U0001F464"          
STAR_ICON = "\u2B50"                

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#E3F2FD,#FFFFFF,#F3E5F5);
}

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#1565C0;
margin-bottom:20px;
}

.metric-card{
background:white;
padding:20px;
border-radius:18px;
box-shadow:0px 4px 15px rgba(0,0,0,.18);
text-align:center;
transition:0.3s;
}

.metric-card:hover{
transform:translateY(-6px);
box-shadow:0px 8px 25px rgba(0,0,0,.25);
}

.metric-number{
font-size:42px;
font-weight:bold;
}

.metric-title{
font-size:20px;
}

.section-title{
font-size:28px;
font-weight:bold;
color:#0D47A1;
padding-top:15px;
padding-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# CHECK SESSION STATE
# ==========================================
current_model = st.session_state.get("selected_model")
prediction_model = st.session_state.get("prediction_model")

if current_model != prediction_model:
    st.warning(
        f"You changed the model to '{current_model}'. "
        "Please generate predictions again."
    )
    st.stop()
if (
    "results" not in st.session_state
    or "prediction_model" not in st.session_state
):
    st.warning("Please generate predictions first.")
    st.stop()

results = st.session_state["results"]


excellent = st.session_state["excellent"]
average = st.session_state["average"]
fail = st.session_state["fail"]
total = st.session_state["total"]

model_name = st.session_state["selected_model"]

# ==========================================
# HEADER
# ==========================================

st.markdown(
f'<div class="title">{CHART_ICON} Prediction Results Dashboard</div>',
unsafe_allow_html=True
)

st.success(
f"Predictions generated successfully using {model_name}"
)
# ==========================================
# MODEL EVALUATION METRICS
# ==========================================
st.markdown("---")

st.markdown(
    f'<div class="section-title">{CHART_ICON} Model Evaluation Metrics</div>',
    unsafe_allow_html=True
)

if "metrics" in st.session_state:

    metrics = st.session_state["metrics"]

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.4f}"
    )

    m2.metric(
        "AUC Score",
        f"{metrics['AUC']:.4f}" if metrics["AUC"] else "N/A"
    )

    m3.metric(
        "Precision",
        f"{metrics['Precision']:.4f}"
    )


    m4, m5, m6 = st.columns(3)

    m4.metric(
        "Recall",
        f"{metrics['Recall']:.4f}"
    )

    m5.metric(
        "F1 Score",
        f"{metrics['F1 Score']:.4f}"
    )

    m6.metric(
        "MCC Score",
        f"{metrics['MCC']:.4f}"
    )

else:

    st.warning(
        "Evaluation metrics not found. Please generate predictions again."
    )

# ==========================================
# SUMMARY CARDS
# ==========================================

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title"><br>Excellent</div>
    <div class="metric-number" style="color:green;">{excellent}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title"><br>Average</div>
    <div class="metric-number" style="color:orange;">{average}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title"><br>Fail</div>
    <div class="metric-number" style="color:red;">{fail}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">{PERSON_ICON}<br>Total Students</div>
    <div class="metric-number" style="color:#1565C0;">{total}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
# ==========================================
# VISUAL ANALYTICS
# ==========================================

st.markdown(
    f'<div class="section-title">{CHART_ICON} Prediction Analytics</div>',
    unsafe_allow_html=True
)

chart_col1, chart_col2 = st.columns(2)

# ==========================================
# BAR CHART
# ==========================================

with chart_col1:

    st.subheader(f"{CHART_ICON} Prediction Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    categories = [
        "Excellent",
        "Average",
        "Fail"
    ]

    values = [
        excellent,
        average,
        fail
    ]

    colors = [
        "#4CAF50",
        "#FFC107",
        "#F44336"
    ]

    bars = ax.bar(
        categories,
        values,
        color=colors,
        edgecolor="black"
    )

    ax.set_ylabel("Number of Students")

    ax.set_title("Student Performance Distribution")

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            int(height),
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    st.pyplot(fig)

# ==========================================
# PIE CHART
# ==========================================

with chart_col2:

    st.subheader(f"{CHART_ICON} Performance Percentage")

    fig2, ax2 = plt.subplots(figsize=(6,5))

    ax2.pie(

        values,

        labels=categories,

        colors=colors,

        autopct="%1.1f%%",

        startangle=90,

        shadow=True

    )

    ax2.axis("equal")

    st.pyplot(fig2)

st.markdown("---")

# ==========================================
# PROGRESS SECTION
# ==========================================

st.markdown(
    f'<div class="section-title">{STAR_ICON} Performance Summary</div>',
    unsafe_allow_html=True
)

excellent_percent = (excellent / total) * 100
average_percent = (average / total) * 100
fail_percent = (fail / total) * 100

st.write(f"**Excellent Students** ({excellent} students)")

st.progress(excellent_percent / 100)

st.caption(f"{excellent_percent:.2f}%")

st.write(f"**Average Students** ({average} students)")

st.progress(average_percent / 100)

st.caption(f"{average_percent:.2f}%")

st.write(f"**Fail Students** ({fail} students)")

st.progress(fail_percent / 100)

st.caption(f"{fail_percent:.2f}%")

st.markdown("---")

# ==========================================
# QUICK STATISTICS
# ==========================================

st.markdown(
    f'<div class="section-title">{SUCCESS_ICON} Quick Statistics</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2)

with left:
    st.info(f"""
### Dataset Summary

- **Total Students:** {total}
- **Model Used:** {model_name}
- **Excellent:** {excellent}
- **Average:** {average}
- **Fail:** {fail}
""")
with right:

    if excellent >= average and excellent >= fail:

        st.success(
            f"{GREEN_ICON} Most students belong to the Excellent category."
        )

    elif average >= excellent and average >= fail:

        st.warning(
            f"{YELLOW_ICON} Most students belong to the Average category."
        )

    else:

        st.error(
            f"{RED_ICON} Most students belong to the Fail category."
        )

st.markdown("---")
# ==========================================
# PREDICTION TABLE
# ==========================================

st.markdown(
    f'<div class="section-title">{TABLE_ICON} Prediction Results</div>',
    unsafe_allow_html=True
)

search = st.text_input(
    "Search Prediction",
    placeholder="Type Excellent, Average or Fail..."
)

display_df = results.copy()

if search:

    display_df = display_df[
        display_df["Prediction"].str.contains(
            search,
            case=False,
            na=False
        )
    ]

def highlight_prediction(row):

    value = row["Prediction"]

    if "Excellent" in value:

        return [
            "background-color:#E8F5E9;color:#1B5E20"
        ] * len(row)

    elif "Average" in value:

        return [
            "background-color:#FFF8E1;color:#E65100"
        ] * len(row)

    else:

        return [
            "background-color:#FFEBEE;color:#B71C1C"
        ] * len(row)

styled_df = display_df.style.apply(
    highlight_prediction,
    axis=1
)

st.dataframe(
    styled_df,
    use_container_width=True,
    height=450
)

st.markdown("---")

# ==========================================
# DOWNLOAD SECTION
# ==========================================

st.markdown(
    f'<div class="section-title">{DOWNLOAD_ICON} Export Predictions</div>',
    unsafe_allow_html=True
)

csv = results.to_csv(
    index=False
).encode("utf-8")

st.download_button(

    label=f"{DOWNLOAD_ICON} Download Predictions CSV",

    data=csv,

    file_name="Student_Performance_Predictions.csv",

    mime="text/csv"

)

st.markdown("---")

# ==========================================
# RESULT MESSAGE
# ==========================================

if excellent >= average and excellent >= fail:

    st.success(
        f"{GREEN_ICON} Excellent is the dominant predicted class."
    )

elif average >= excellent and average >= fail:

    st.info(
        f"{YELLOW_ICON} Average is the dominant predicted class."
    )

else:

    st.warning(
        f"{RED_ICON} Fail is the dominant predicted class."
    )

st.markdown("---")

# ==========================================
# CELEBRATION
# ==========================================



st.success(
    f"Prediction process completed successfully!"
)

st.markdown("---")

# ==========================================
# FOOTER
# ==========================================

st.markdown(
"""
<center>

<h4 style='color:#1565C0;'>

 Student Performance Classification Dashboard

</h4>

Developed using

<b>Streamlit • Scikit-learn • Pandas • Matplotlib</b>

</center>
""",
unsafe_allow_html=True
)


# ==========================================
# BACK BUTTON
# ==========================================

if st.button(f"{BACK_ICON} Back to Home"):

    st.switch_page("streamlit_app.py")