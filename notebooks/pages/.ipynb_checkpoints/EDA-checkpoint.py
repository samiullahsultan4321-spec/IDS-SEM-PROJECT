import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
st.set_page_config(
    page_title="EDA – Car Price Analysis",
    layout="wide"
)
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_data.csv")
    df.columns = df.columns.astype(str).str.lower()
    cur_year = datetime.now().year
    df['car_age'] = cur_year - df['prod._year']
    #df = df.drop(['id','prod._year'],axis=1)
    return df

df = load_data()

st.header("Exploratory Data Analysis (EDA)")

# Dataset preview
st.subheader("Dataset Preview")
if st.checkbox("Show Dataset Table"):
    st.dataframe(df.head(10))

# Missing values
st.subheader("Missing Values Summary")
missing_value_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Count": df.isnull().sum(),
    "Percent Missing": df.isnull().sum() / len(df) * 100
})
if st.checkbox('Show Missing Values Summary'):
    st.dataframe(missing_value_df)

# Price distribution
st.subheader("Price Distribution")
fig1, ax = plt.subplots()
sns.histplot(df['price'], bins=30, kde=True, ax=ax)
if st.checkbox('Show Price Distribution'):
    st.pyplot(fig1)
    plt.close(fig1)
# Price vs Mileage
st.subheader("Price vs Mileage")
fig2, ax = plt.subplots()
sns.scatterplot(x=df['mileage'], y=df['price'], ax=ax)
if st.checkbox('Show Price vs Mileage'):
    st.pyplot(fig2)
    plt.close(fig2)
# Price vs Car Age
st.subheader("Price vs Car Age")
fig3, ax = plt.subplots()
sns.scatterplot(x=df['car_age'], y=df['price'], ax=ax)
if st.checkbox('Show Price vs Car Age'):
    st.pyplot(fig3)
    plt.close(fig3)
