import streamlit as st
st.set_page_config(
    page_title="Introduction ",
    layout="wide"
)

st.title("🚗 Car Price Prediction – Analysis & ML Model")
st.sidebar.success("Select a page from above 👆")

st.header("Introduction")
st.write("""
Welcome to the Used Car Price Prediction app!  

This project analyzes a dataset of used car listings to understand the factors affecting car prices 
and builds a machine learning model to predict the selling price of a car based on its features.

**Dataset Overview:**
- Features include engine volume, mileage, car age, fuel type, category, airbags, and more.
- The target variable is the car's price in the local market.

**Project Goal:**
- Perform Exploratory Data Analysis (EDA) to extract insights from the dataset.
- Preprocess categorical and numerical features using encoding and scaling techniques.
- Train a Random Forest Regressor to predict car prices accurately.
- Provide an interactive interface for real-time price prediction.

This app allows users to input the specifications of a car and receive an instant price prediction based on the trained model.
""")
