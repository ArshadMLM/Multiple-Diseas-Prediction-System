# -*- coding: utf-8 -*-
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# =========================
# LOAD MODELS & SCALERS
# =========================
diabetes_model = pickle.load(open('C:/Desktop/Multiple Disease Prediction System/diabetes_model.sav', 'rb'))
diabetes_scaler = pickle.load(open('C:/Desktop/Multiple Disease Prediction System/diabetes_scaler.sav', 'rb'))

heart_model = pickle.load(open('C:/Desktop/Multiple Disease Prediction System/heart_disease_model.sav', 'rb'))
heart_scaler = pickle.load(open('C:/Desktop/Multiple Disease Prediction System/heart_scaler.sav', 'rb'))

parkinson_model = pickle.load(open('C:/Desktop/Multiple Disease Prediction System/parkinsons_model.sav', 'rb'))
parkinson_scaler = pickle.load(open('C:/Desktop/Multiple Disease Prediction System/parkinsons_scaler.sav', 'rb'))

breast_cancer_model = pickle.load(open('C:/Desktop/Multiple Disease Prediction System/breast_cancer_model.sav', 'rb'))
breast_cancer_scaler = pickle.load(open('C:/Desktop/Multiple Disease Prediction System/breast_cancer_scaler.sav', 'rb'))

# =========================
# SIDEBAR MENU
# =========================
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson Prediction', 'Breast Cancer Prediction'],
        icons=['activity','heart','person','gender-female'],
        default_index=0
    )

# ======================================================
# 🩸 DIABETES PAGE
# ======================================================
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('No of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness')
        DPF = st.text_input('Diabetes Pedigree Function')

    with col2:
        Glucose = st.text_input('Glucose level')
        Insulin = st.text_input('Insulin level')
        Age = st.text_input('Age')

    with col3:
        BloodPressure = st.text_input('Blood Pressure')
        BMI = st.text_input('BMI')

    if st.button('Diabetes Test Result'):
        if '' in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]:
            st.warning('Please enter all values')
        else:
            input_data = np.array([[Pregnancies, Glucose, BloodPressure,
                                    SkinThickness, Insulin, BMI, DPF, Age]], dtype=float)
            scaled_data = diabetes_scaler.transform(input_data)
            prediction = diabetes_model.predict(scaled_data)

            if prediction[0] == 0:
                st.success(
                    "🌟 Good news! No signs of diabetes detected.\n\n"
                    "Your health looks good 💚\n"
                    "Continue eating healthy 🥗, staying active 🚶‍♂️,\n"
                    "and taking care of yourself every day 😊"
                )
            else:
                st.warning(
                    "💛 Diabetes indicators detected — please don’t lose hope.\n\n"
                    "Diabetes is a manageable condition.\n"
                    "With proper diet 🥗, regular exercise 🏃‍♀️,\n"
                    "and guidance from a doctor 👨‍⚕️,\n"
                    "you can live a normal, healthy, and happy life 🌈"
                )

# ======================================================
# ❤️ HEART DISEASE PAGE
# ======================================================
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
        trestbps = st.text_input('Resting Blood Pressure')
        restecg = st.text_input('Resting ECG results')
        oldpeak = st.text_input('ST depression')
        thal = st.text_input('Thal')

    with col2:
        sex = st.text_input('Sex (1 = Male, 0 = Female)')
        chol = st.text_input('Serum Cholesterol')
        thalach = st.text_input('Max Heart Rate Achieved')
        slope = st.text_input('Slope of peak exercise ST segment')

    with col3:
        cp = st.text_input('Chest Pain Type')
        fbs = st.text_input('Fasting Blood Sugar (1 = Yes, 0 = No)')
        exang = st.text_input('Ex.Induced Angina (1 = Yes, 0 = No)')
        ca = st.text_input('Major vessels')
        
    if st.button('Heart Disease Test Result'):
        if '' in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]:
            st.warning('Please enter all values')
        else:
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                    restecg, thalach, exang, oldpeak,
                                    slope, ca, thal]], dtype=float)
            scaled_data = heart_scaler.transform(input_data)
            prediction = heart_model.predict(scaled_data)

            if prediction[0] == 0:
                st.success(
                    "❤️ Your heart shows no signs of disease.\n\n"
                    "That’s reassuring news 💚\n"
                    "Maintain a heart-healthy lifestyle 🥗🏃‍♂️\n"
                    "and keep smiling 😊"
                )
            else:
                st.warning(
                    "💙 Heart-related indicators detected.\n\n"
                    "Please remember — early awareness is powerful.\n"
                    "With lifestyle changes 🌱 and proper medical care 👨‍⚕️,\n"
                    "heart conditions can be well managed.\n"
                    "You are not alone 💪"
                )

# ======================================================
# 🧠 PARKINSON PAGE
# ======================================================
elif selected == 'Parkinson Prediction':
    st.title('Parkinson Prediction using ML (Top 20 features)')

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        jitter_percent = st.text_input('MDVP:Jitter(%)')
        rap = st.text_input('MDVP:RAP')
        shimmer = st.text_input('MDVP:Shimmer')
        nhr = st.text_input('NHR')
        rpde = st.text_input('RPDE')
        spread1 = st.text_input('spread1')


    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        jitter_abs = st.text_input('MDVP:Jitter(Abs)')
        ppq = st.text_input('MDVP:PPQ')
        shimmer_db = st.text_input('MDVP:Shimmer(dB)')
        hnr = st.text_input('HNR')
        dfa = st.text_input('DFA')
        spread2 = st.text_input('spread2')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        ddp = st.text_input('Jitter:DDP')
        apq3 = st.text_input('Shimmer:APQ3')
        apq5 = st.text_input('Shimmer:APQ5')
        apq = st.text_input('MDVP:APQ')
        dda = st.text_input('Shimmer:DDA')


    # Prepare input fields (20 features)
    input_fields = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                    shimmer, shimmer_db, apq3, apq5, apq, dda,
                    nhr, hnr, rpde, dfa, spread1, spread2]

    if st.button('Parkinson Test Result'):
        if '' in input_fields:
            st.warning('Please enter all values')
        else:
            input_data = np.array([list(map(float, input_fields))])
            scaled_data = parkinson_scaler.transform(input_data)
            prediction = parkinson_model.predict(scaled_data)

            if prediction[0] == 0:
                st.success(
                    "🧠 No signs of Parkinson’s disease detected.\n\n"
                    "That’s comforting news 💚\n"
                    "Stay mentally and physically active,\n"
                    "and continue caring for your well-being 😊"
                )
            else:
                st.warning(
                    "💛 Parkinson-related indicators detected.\n\n"
                    "Please don’t lose hope.\n"
                    "With early medical care, therapy, and support 👨‍⚕️,\n"
                    "quality of life can be maintained 🌈"
                )


# ======================================================
# 🎗️ BREAST CANCER PAGE
# ======================================================
elif selected == 'Breast Cancer Prediction':
    st.title('Breast Cancer Prediction using ML')

    col1, col2 = st.columns(2)

    with col1:
        radius_mean = st.text_input('Radius Mean')
        perimeter_mean = st.text_input('Perimeter Mean')
        smoothness_mean = st.text_input('Smoothness Mean')
        concavity_mean = st.text_input('Concavity Mean')
        symmetry_mean = st.text_input('Symmetry Mean')

    with col2:
        texture_mean = st.text_input('Texture Mean')
        area_mean = st.text_input('Area Mean')
        compactness_mean = st.text_input('Compactness Mean')
        concave_points_mean = st.text_input('Concave Points Mean')
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean')

    input_fields = [radius_mean, texture_mean, perimeter_mean, area_mean,
                    smoothness_mean, compactness_mean, concavity_mean,
                    concave_points_mean, symmetry_mean, fractal_dimension_mean]

    if st.button('Breast Cancer Test Result'):
        if '' in input_fields:
            st.warning('Please enter all values')
        else:
            input_data = np.array([list(map(float, input_fields))])
            scaled_data = breast_cancer_scaler.transform(input_data)
            prediction = breast_cancer_model.predict(scaled_data)

            if prediction[0] == 0:
                st.warning(
                    "🎗️ Malignant indicators detected.\n\n"
                    "Please remember — early detection saves lives.\n"
                    "With timely treatment, medical care 👨‍⚕️,\n"
                    "and strong emotional support 💪,\n"
                    "many people recover and live healthy lives 🌸"
                )
            else:
                st.success(
                    "🌷 Benign tumor detected.\n\n"
                    "That’s reassuring news.\n"
                    "Continue regular check-ups 🩺 and self-care,\n"
                    "and stay confident about your health 😊"
                )
