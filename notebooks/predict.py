import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from preprocessing import apply_label_encoding, apply_target_encoding,fit_label_encoders,fit_target_encoders,get_preprocessor

def predict(new_car):
        
    label_encoders = joblib.load(r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\label_encoders.pkl")
    target_encoders = joblib.load(r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\target_encoders.pkl")
    ml_model = joblib.load(r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\model.pkl")
    preprocessor = joblib.load(r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\preprocessor.pkl")
    global_mean = joblib.load(r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\global_mean.pkl")
    new_car = apply_label_encoding(new_car, label_encoders)
    new_car = apply_target_encoding(new_car, target_encoders,global_mean)
    new_car_transformed = preprocessor.transform(new_car)
    predicted_price = ml_model.predict(new_car_transformed)
    print("Predicted Car Price:", predicted_price[0])

new_car1 = pd.DataFrame([{
    'manufacturer': 'Suzuki',
    'model': 'Alto',
    'leather_interior': False,
    'wheel': 'Left wheel',
    'fuel_type': 'Petrol',
    'gear_box_type': 'Manual',
    'drive_wheels': 'FWD',
    'doors': 4,
    'color': 'White',
    'category': 'Hatchback',
    'levy': 500,
    'engine_volume': 800,
    'mileage': 150000,
    'cylinders': 3,
    'car_age': 10,
    'airbags': 2}])
new_car2 = pd.DataFrame([{
    'manufacturer': 'Honda',
    'model': 'Civic',
    'leather_interior': True,
    'wheel': 'Left wheel',
    'fuel_type': 'Petrol',
    'gear_box_type': 'Automatic',
    'drive_wheels': 'FWD',
    'doors': 4,
    'color': 'Black',
    'category': 'Sedan',
    'levy': 2000,
    'engine_volume': 1800,
    'mileage': 40000,
    'cylinders': 4,
    'car_age': 4,
    'airbags': 6}])
new_car3 = pd.DataFrame([{
    'manufacturer': 'Toyota',
    'model': 'Fortuner',
    'leather_interior': True,
    'wheel': 'Left wheel',
    'fuel_type': 'Diesel',
    'gear_box_type': 'Automatic',
    'drive_wheels': '4WD',
    'doors': 5,
    'color': 'Silver',
    'category': 'SUV',
    'levy': 5000,
    'engine_volume': 2700,
    'mileage': 30000,
    'cylinders': 6,
    'car_age': 3,
    'airbags': 8}])
new_car4 = pd.DataFrame([{
    'manufacturer': 'BMW',
    'model': 'M4',
    'leather_interior': True,
    'wheel': 'Left wheel',
    'fuel_type': 'Petrol',
    'gear_box_type': 'Automatic',
    'drive_wheels': 'RWD',
    'doors': 2,
    'color': 'Red',
    'category': 'Coupe',
    'levy': 15000,
    'engine_volume': 3000,
    'mileage': 5000,
    'cylinders': 6,
    'car_age': 1,
    'airbags': 6}])

new_car5 = pd.DataFrame([{
    'manufacturer': 'Hyundai',
    'model': 'Santro Classic',
    'leather_interior': False,
    'wheel': 'Left wheel',
    'fuel_type': 'Petrol',
    'gear_box_type': 'Manual',
    'drive_wheels': 'FWD',
    'doors': 4,
    'color': 'Blue',
    'category': 'Hatchback',
    'levy': 1000,
    'engine_volume': 1000,
    'mileage': 200000,
    'cylinders': 4,
    'car_age': 15,
    'airbags': 2}])

predict(new_car1)
predict(new_car2)
predict(new_car3)
predict(new_car4)
predict(new_car5)