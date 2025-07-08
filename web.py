import streamlit as st
import pandas as pd
import joblib
import numpy as np
k = joblib.load("linear_model.pkl")
st.title("Sleep Predictor")
WorkoutTime= st.text_input("Enter your workout time per day")
ReadingTime= st.text_input("Enter your reading time per day")
PhoneTime= st.text_input("Enter your phone time per day")
WorkHours= st.text_input("Enter the amount of hours worked per day")
RelaxationTime= st.text_input("Enter your relaxation time per day")

if st.button("Predict"):
   
    try:
      X_input = pd.DataFrame([{
        "WorkoutTime": float(WorkoutTime),
        "ReadingTime": float(ReadingTime),
        "PhoneTime": float(PhoneTime),
        "WorkHours": float(WorkHours),
        "RelaxationTime": float(RelaxationTime)         
      }])
      sleep = np.ceil(k.predict(X_input)[0])
      st.subheader("🔮 Predicted Sleep")
      st.success(f"{sleep} hours")
    
    except ValueError:
        st.error("Please enter valid numeric values in all fields.")
        
    
