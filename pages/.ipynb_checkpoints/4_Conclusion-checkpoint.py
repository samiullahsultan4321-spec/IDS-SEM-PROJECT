import streamlit as st

st.set_page_config(page_title="Results & Conclusions", layout="wide")

st.header("📊 Results and Conclusions")

st.write("""
This page summarizes the key findings from the EDA and the ML model predictions:

### Key Observations from EDA:
- Price generally decreases with car age and high mileage.
- Certain manufacturers and car categories have consistently higher prices.
- Fuel type, gear type, and drive wheels influence price moderately.

### ML Model Performance:
- Model used: Random Forest Regressor
- Preprocessing: Label Encoding for categorical variables, Target Encoding for high-cardinality features
- Accuracy: R² score = 0.65, RMSE = 11060.75

### Conclusions:
1. Used car prices are strongly affected by age, mileage, and engine volume.
2. Premium brands and rare categories tend to retain higher value.
3. The trained ML model provides reasonable predictions and can be used as a tool for buyers and sellers.

### Future Improvements:
- Collect more diverse data for underrepresented car types.
- Incorporate additional features like service history, accidents, and ownership.
- Test other ML models like Gradient Boosting or XGBoost for higher accuracy.
""")
