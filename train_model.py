import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = {
    'car_age': [3, 5, 2, 7, 4, 6, 1, 8, 3, 5, 2, 4, 3, 5, 2, 6, 1, 4],
    'km_driven': [30000, 50000, 15000, 80000, 40000, 65000, 10000, 90000, 25000, 55000, 18000, 35000, 20000, 45000, 12000, 60000, 8000, 30000],
    'engine_capacity': [1200, 1500, 1000, 1800, 1200, 1400, 1200, 1600, 1000, 1500, 1200, 2000, 2500, 3000, 2000, 1500, 1200, 2200],
    'mileage': [18.5, 15.0, 20.0, 12.0, 19.0, 16.0, 21.0, 11.5, 20.5, 14.5, 19.5, 17.5, 14.0, 12.0, 15.0, 18.0, 25.0, 16.0],
    'previous_owners': [1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1],
    'fuel_type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'CNG', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Electric', 'Diesel'],
    'transmission': ['Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Automatic', 'Automatic', 'Automatic', 'Manual', 'Automatic', 'Automatic'],
    'brand': ['Maruti', 'Hyundai', 'Maruti', 'Toyota', 'Honda', 'Hyundai', 'Maruti', 'Toyota', 'Maruti', 'Hyundai', 'Tata', 'Mahindra', 'BMW', 'Mercedes', 'Audi', 'Tata', 'Tata', 'Mahindra'],
    'selling_price': [450000, 550000, 500000, 400000, 480000, 520000, 600000, 350000, 420000, 530000, 520000, 750000, 2800000, 3500000, 3000000, 600000, 1100000, 950000]
}

df = pd.DataFrame(data)

# Separate features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

categorical_cols = ['fuel_type', 'transmission', 'brand']


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print("--- Model Evaluation Metrics ---")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

joblib.dump(model_pipeline, 'car_price_model.pkl')
print("Model trained and saved successfully as 'car_price_model.pkl'")