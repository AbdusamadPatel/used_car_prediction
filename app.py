import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            brand TEXT,
            car_age INTEGER,
            km_driven INTEGER,
            engine_capacity INTEGER,
            mileage REAL,
            previous_owners INTEGER,
            fuel_type TEXT,
            transmission TEXT,
            predicted_price REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(brand, car_age, km_driven, engine_capacity, mileage, previous_owners, fuel_type, transmission, predicted_price):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO predictions (timestamp, brand, car_age, km_driven, engine_capacity, mileage, previous_owners, fuel_type, transmission, predicted_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, brand, car_age, km_driven, engine_capacity, mileage, previous_owners, fuel_type, transmission, predicted_price))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

init_db()

st.title("🚗 Used Car Price Prediction System")

try:
    model = joblib.load('car_price_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run 'train_model.py' first!")

st.write("Enter the car details below to predict estimated market selling price:")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Car Brand", ['Maruti', 'Hyundai', 'Toyota', 'Honda', 'Tata', 'Mahindra', 'BMW', 'Mercedes', 'Audi'])
    car_age = st.number_input("Car Age (Years)", min_value=0, max_value=30, value=3)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000)
    engine_capacity = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, value=1200)

with col2:
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=18.5)
    previous_owners = st.selectbox("Number of Previous Owners", [1, 2, 3, 4])
    fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'])
    transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])

if st.button("Predict Price", type="primary"):
    input_data = pd.DataFrame([{
        'car_age': car_age,
        'km_driven': km_driven,
        'engine_capacity': engine_capacity,
        'mileage': mileage,
        'previous_owners': previous_owners,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'brand': brand
    }])

    st.subheader("📋 Input Summary")
    st.dataframe(input_data)

    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)

    save_prediction(brand, car_age, km_driven, engine_capacity, mileage, previous_owners, fuel_type, transmission, prediction)

    st.success(f"💰 **Predicted Selling Price:** ₹{prediction:,.2f}")
    st.info("Data successfully saved to database (`predictions.db`)!")

with st.expander("📜 View Prediction History (Database)"):
    history_df = get_history()
    if not history_df.empty:
        st.dataframe(history_df)
    else:
        st.write("No prediction history found.")