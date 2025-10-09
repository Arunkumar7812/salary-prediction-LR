import streamlit as st
import pickle
import numpy as np
import os

# --- Model Loading ---
# Based on the provided Jupyter notebook, the model is likely saved as
# 'salary_prediction.pkl' and contains a single LinearRegression model.
MODEL_FILE = 'salary_prediction.pkl'

try:
    with open(MODEL_FILE, 'rb') as f:
        # Load the model. We assume only the model object was saved.
        model = pickle.load(f)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(f"Model file '{MODEL_FILE}' not found. Please ensure it is uploaded.")
except Exception as e:
    model_loaded = False
    st.error(f"Error loading the model: {e}")

st.title("💸 Salary Prediction App")
st.write("This application estimates salary based on the 'Years of Experience' feature, using the model saved in your provided notebook.")

if model_loaded:
    st.subheader("Enter Employee Details")

    # The model from the notebook uses 'Years of Experience' as the single feature.
    years_of_experience = st.number_input(
        "Years of Experience:",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=0.1,
        help="Input the number of years the employee has worked."
    )

    # --- Prediction Logic ---
    if st.button("Predict Salary", type="primary"):
        # The model expects a 2D array: [[feature_value]]
        features = np.array([[years_of_experience]])

        try:
            # Predict the salary
            prediction = model.predict(features)

            # Display the result formatted as currency
            estimated_salary = prediction[0]
            st.success(f"Estimated Salary: **${estimated_salary:,.2f}**")
            st.balloons()
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    st.markdown("---")
    st.caption(f"Model used: Simple Linear Regression loaded from **{MODEL_FILE}**.")
else:
    st.warning("Prediction functionality is disabled until the model file is successfully loaded.")
