import pandas as pd
import joblib
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path("notebooks").resolve()))
from preprocessing import apply_label_encoding, apply_target_encoding,fit_label_encoders,fit_target_encoders,get_preprocessor

st.set_page_config(
    page_title="ML Prediction Model",
    layout="wide"
)
@st.cache_resource
def load_model():
    label_encoders = joblib.load("encoders/label_encoders.pkl")
    target_encoders = joblib.load("encoders/target_encoders.pkl")
    ml_model = joblib.load("encoders/model.pkl")
    preprocessor = joblib.load("encoders/preprocessor.pkl")
    global_mean = joblib.load("encoders/global_mean.pkl")
    return label_encoders, target_encoders, ml_model, preprocessor, global_mean

def prediction():
    manufacturer = st.text_input("Manufacturer", "HONDA")
    model_name = st.text_input("Model", "Civic")
    
    leather = st.selectbox("Leather Interior", ["True", "False"])
    wheel = st.selectbox("Wheel", ["Left wheel", "Right-hand drive"])
    
    fuel = st.selectbox("Fuel Type", ['Hybrid', 'Petrol', 'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG','Hydrogen'])
    gear = st.selectbox("Gear Box Type", ['Automatic', 'Tiptronic', 'Variator', 'Manual'])
    drive = st.selectbox("Drive Wheels", ['4x4', 'Front', 'Rear'])
    doors = st.selectbox("Doors", [2, 4, 6])
    color = st.selectbox("Color", ['Silver', 'Black', 'White', 'Grey', 'Blue', 'Green', 'Red','Sky blue', 'Orange', 'Yellow', 'Brown', 'Golden', 'Beige', 'Carnelian red', 'Purple', 'Pink'])
    category = st.selectbox("Category", ['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon','Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine','Pickup'])
    
    levy = st.number_input("Levy", 0, 5000, 2500)
    engine = st.number_input("Engine Volume (cc)", 1.00, 6.50, 3.00)
    mileage = st.number_input("Mileage", 0, 571000, 50000)
    cylinders = st.number_input("Cylinders", 1, 16, 4)
    car_age = st.number_input("Car Age (years)", 0, 90, 5)
    airbags = st.number_input("Airbags", 0, 16, 6)
    
    price = None
    if st.button("Predict Price"):
    
        input_df = pd.DataFrame([{
            "manufacturer": manufacturer,
            "model": model_name,
            "leather_interior": leather,
            "wheel": wheel,
            "fuel_type": fuel,
            "gear_box_type": gear,
            "drive_wheels": drive,
            "doors": doors,
            "color": color,
            "category": category,
            "levy": levy,
            "engine_volume": engine,
            "mileage": mileage,
            "cylinders": cylinders,
            "car_age": car_age,
            "airbags": airbags
        }])
        label_encoders, target_encoders, ml_model, preprocessor, global_mean = load_model()
        input_df = apply_label_encoding(input_df, label_encoders)
        input_df = apply_target_encoding(input_df, target_encoders,global_mean)
        
        # Preprocessing
        input_df = preprocessor.transform(input_df)
        
        # Prediction
        price = ml_model.predict(input_df)[0]
    return price

def main():
    st.header('ML Prediction Model')
    st.write("""
    This section allows users to enter car features and get a real-time price prediction
    using a trained Random Forest regression model.
        """)
    
    st.subheader("Enter Car Details")
    price = prediction()
    if price is not None:
        st.success(f"💰 Predicted Car Price: {price:,.2f}")
    
if __name__ == "__main__":
    main()

