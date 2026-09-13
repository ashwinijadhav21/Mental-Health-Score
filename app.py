import streamlit as st
import pandas as pd
import joblib
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="🧠",
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_FILE = "Mental_Health_Model.pkl"

if not os.path.exists(MODEL_FILE):
    st.error(
        f"Model file '{MODEL_FILE}' was not found. "
        "Please keep the .pkl file in the same folder as app.py."
    )
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# =========================================================
# COUNTRY GROUPING
# =========================================================

top_countries = [
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France"
]


# =========================================================
# TITLE
# =========================================================

st.title("🧠 Mental Health Prediction")
st.write("Enter student information to predict the mental health score.")


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("Student Information")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=21,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    country = st.text_input(
        "Country",
        value="India"
    )

    academic_level = st.selectbox(
        "Academic Level",
        [
            "Undergraduate",
            "Graduate",
            "High School"
        ]
    )

    most_used_platform = st.selectbox(
        "Most Used Platform",
        [
            "Facebook",
            "LinkedIn",
            "Instagram",
            "Snapchat",
            "Twitter",
            "YouTube",
            "TikTok",
            "LINE",
            "KakaoTalk",
            "VKontakte",
            "WhatsApp",
            "WeChat"
        ]
    )

    purpose_of_use = st.selectbox(
        "Purpose of Use",
        [
            "Networking",
            "Education",
            "Entertainment",
            "News"
        ]
    )


with col2:

    avg_daily_usage_hours = st.number_input(
        "Average Daily Usage Hours",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5
    )

    daily_unlocks = st.number_input(
        "Daily Unlocks",
        min_value=0,
        value=50,
        step=1
    )

    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    physical_activity_hours = st.number_input(
        "Physical Activity Hours",
        min_value=0.0,
        max_value=24.0,
        value=1.0,
        step=0.5
    )

    sleep_hours_per_night = st.number_input(
        "Sleep Hours Per Night",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    stress_level = st.selectbox(
        "Stress Level",
        [
            "Medium",
            "Low",
            "Very High",
            "High"
        ]
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔮 Predict Mental Health Score",
    use_container_width=True
):

    # -----------------------------------------------------
    # GROUP COUNTRY
    # -----------------------------------------------------

    if country in top_countries:
        grouped_country = country
    else:
        grouped_country = "Other"


    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # IMPORTANT:
    # These column names MUST match the trained model.
    # -----------------------------------------------------

    input_data = pd.DataFrame([{

        "Study_Hours": study_hours,

        "Age": age,

        "Avg_Daily_Usage_Hours": avg_daily_usage_hours,

        "Daily_Unlocks": daily_unlocks,

        "Physical_Activity_Hours": physical_activity_hours,

        "Sleep_Hours_Per_Night": sleep_hours_per_night,

        "Stress_Level": stress_level,

        "Gender": gender,

        "grouped_country": grouped_country,

        "Academic_Level": academic_level,

        "Most_Used_Platform": most_used_platform,

        "Purpose_Of_Use": purpose_of_use

    }])


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        prediction = model.predict(input_data)

        score = float(prediction[0])

        # Keep score inside expected range if necessary
        score = max(0, min(10, score))


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted Mental Health Score",
            value=f"{score:.2f}"
        )


        # -------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------

        if score < 4:

            st.info(
                "The predicted score is relatively low."
            )

        elif score < 7:

            st.warning(
                "The predicted score is in the moderate range."
            )

        else:

            st.success(
                "The predicted score is relatively high."
            )


        # -------------------------------------------------
        # DEBUG INFORMATION
        # -------------------------------------------------

        with st.expander("View Input Data"):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

        st.write("Columns sent to the model:")

        st.write(
            list(input_data.columns)
        )