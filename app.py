import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
import sklearn

DATASET_PATH = "data/ISP DATA HEART DISEASES.csv"
LOG_MODEL_PATH = "model/LR_model.pkl"


def main():
    @st.cache(persist=True)
    def load_dataset() -> pd.DataFrame:
        heart_df = pl.read_csv(DATASET_PATH)
        heart_df = heart_df.to_pandas()
        heart_df = pd.DataFrame(np.sort(heart_df.values, axis=0),
                                index=heart_df.index,
                                columns=heart_df.columns)
        return heart_df


    def user_input_features() -> pd.DataFrame:
        Race = st.sidebar.selectbox("Race", options=(race for race in heart.Race.unique()))
        Sex = st.sidebar.selectbox("Sex", options=(sex for sex in heart.Sex.unique()))
        AgeCategory = st.sidebar.selectbox("Age category",
                                       options=(AgeCategory for AgeCategory in heart.AgeCategory.unique()))
        BMI = st.sidebar.selectbox("BMI category",
                                       options=(BMI for BMI in heart.BMICategory.unique()))
        SleepTime = st.sidebar.number_input("How many hours on average do you sleep?", 0, 24, 7)
        GenHealth = st.sidebar.selectbox("How can you define your general health?",
                                          options=(gen_health for gen_health in heart.GenHealth.unique()))
        PhysicalActivity = st.sidebar.selectbox("Have you played any sports (running, biking, etc.)"
                                        " in the past month?", options=("No", "Yes"))
        Smoking = st.sidebar.selectbox("Have you smoked at least 100 cigarettes in"
                                       " your entire life (approx. 5 packs)?)",
                                       options=("No", "Yes"))
        AlcoholDrinking = st.sidebar.selectbox("Do you have more than 14 drinks of alcohol (men)"
                                             " or more than 7 (women) in a week?", options=("No", "Yes"))
        Asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))

        features = pd.DataFrame({
            "SleepTime": [SleepTime],
            "BMICategory": [BMI],
            "Smoking": [Smoking],
            "AlcoholDrinking": [AlcoholDrinking],
            "Sex": [Sex],
            "AgeCategory": [AgeCategory],
            "Race": [Race],
            "PhysicalActivity": [PhysicalActivity],
            "GenHealth": [GenHealth],
            "Asthma": [Asthma],

        })

        return features


    st.set_page_config(
        page_title="Heart Disease Prediction Web Application",
        page_icon="images/heart-fav.png"
    )

    st.title("Heart Disease Prediction")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This application will help you to diagnose it!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor.png",
                 caption="I'll help you diagnose your heart health! - Dr. Machine Learning",
                 width=150)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!
        
        Here, a logistic regression model using an undersampling technique
        was constructed using survey data of over 300k US residents from the year 2020.
        This application is based on it because it has proven to be better than the random forest
        (it achieves an accuracy of about 80%, which is quite good).
        
        To predict your heart disease status, simply follow the steps bellow:
        1. Enter the parameters that best describe you;
        2. Press the "Predict" button and wait for the result.
            
        **Keep in mind that this results is not equivalent to a medical diagnosis!
        This model would never be adopted by health care facilities because of its less
        than perfect accuracy, so if you have any problems, consult a human doctor.** """)
        


    heart = load_dataset()

    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=100)

    input_df = user_input_features()
    df = pd.concat([input_df, heart], axis=0)
    df = df.drop(columns=["HeartDisease"])

    cat_cols = ["BMICategory", "Smoking", "AlcoholDrinking",
                "Sex", "AgeCategory", "Race", "PhysicalActivity",
                "GenHealth", "Asthma"]
    for cat_col in cat_cols:
        dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
        df = pd.concat([df, dummy_col], axis=1)
        del df[cat_col]

    df = df[:1]
    df.fillna(0, inplace=True)

    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    if submit:
        prediction = log_model.predict(df)
        prediction_prob = log_model.predict_proba(df)
        if prediction == 0:
            st.markdown(f"**The probability that you'll have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" You are healthy!**")
            st.image("images/heart-okay.jpg",
                     caption="Your heart seems to be okay! - Dr. Machine Learning")
        else:
            st.markdown(f"**The probability that you will have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
            st.image("images/heart-bad.jpg",
                     caption="I'm not satisfied with the condition of your heart! - Dr. Machine Learning")


if __name__ == "__main__":
    main()
