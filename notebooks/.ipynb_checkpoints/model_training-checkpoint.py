import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from preprocessing import apply_label_encoding, apply_target_encoding,fit_label_encoders,fit_target_encoders,get_preprocessor
df1 = pd.read_csv(r"C:\Users\User\Downloads\IDS-SEM-PROJECT\data\cleaned_data.csv")
df1.columns=df1.columns.astype(str).str.lower()


df = df1.copy(deep=True)
df['car_age'] = 2025 - df['prod._year']
df = df.drop(['id','prod._year'],axis=1)

x = df.drop(['price'],axis=1)
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
global_mean = y_train.mean()
label_encoders = fit_label_encoders(x_train)
x_train = apply_label_encoding(x_train, label_encoders)

target_encoders = fit_target_encoders(pd.concat([x_train,y_train],axis=1),'price')
x_train = apply_target_encoding(x_train,target_encoders,global_mean )

preprocessor = get_preprocessor()
x_train = preprocessor.fit_transform(x_train)

x_test = apply_label_encoding(x_test, label_encoders)
x_test = apply_target_encoding(x_test, target_encoders,global_mean)
x_test = preprocessor.transform(x_test)
print('spliting successful!')
ml_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
ml_model.fit(x_train, y_train)

pred = ml_model.predict(x_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
rmse = np.sqrt(mse)
print("Random Forest Results:")
print("RMSE:", rmse)
print("R² Score:", r2)

# Save label encoders
joblib.dump(label_encoders, r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\label_encoders.pkl")
joblib.dump(target_encoders, r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\target_encoders.pkl")
joblib.dump(preprocessor, r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\preprocessor.pkl")
joblib.dump(x_test, r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\x_test.pkl")
joblib.dump(x_train, r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\x_train.pkl")
joblib.dump(y_train, r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\y_train.pkl")
joblib.dump(y_test, r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\y_test.pkl")
joblib.dump(ml_model, r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\model.pkl")
joblib.dump(global_mean, r"C:\Users\User\Downloads\IDS-SEM-PROJECT\encoders\global_mean.pkl")


