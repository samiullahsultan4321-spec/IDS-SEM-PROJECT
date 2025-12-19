import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

label_cols = ['leather_interior','wheel']
cat_cols = ['fuel_type','gear_box_type','drive_wheels','doors','color','category']
scale_cols = ['levy','engine_volume','mileage','cylinders','car_age','airbags']
#freq_cols = ['model','manufacturer']
target_cols = ['model','manufacturer']
def fit_label_encoders(df):
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders
def apply_label_encoding(df, encoders):
    df = df.copy()
    for col in label_cols:
        df[col] = encoders[col].transform(df[col].astype(str))
    return df

# def fit_frequency_encoders(df):
#     """
#     Learn frequency mappings from training data
#     """
#     freq_maps = {}
#     for col in freq_cols:
#         freq_maps[col] = df[col].value_counts().to_dict()
#     return freq_maps


# def apply_frequency_encoding(df, freq_maps):
#     """
#     Apply learned frequency mappings
#     """
#     df = df.copy()
#     for col, mapping in freq_maps.items():
#         df[col] = df[col].map(mapping).fillna(0)
#     return df

def fit_target_encoders(df,target):
    encoders = {}
    for col in target_cols:
        encoders[col] = df.groupby(col)[target].mean().to_dict()
    return encoders

def apply_target_encoding(df, encoders, global_mean):
    df = df.copy()
    for col in target_cols:
        df[col] = df[col].map(encoders[col]).fillna(global_mean)
    return df

def get_preprocessor():
    """
    OneHotEncoding + Scaling
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), scale_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor

