import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st
import os

st.title("🚗 Car Price Prediction")

# Check if model.pkl exists and load it
try:
    with open('model.pkl', 'rb') as f:
        model = pk.load(f)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Load car data
try:
    cars_data = pd.read_csv('Cardetails.csv')
except Exception as e:
    st.error(f"❌ Error loading Cardetails.csv: {e}")
    st.stop()

# Preprocessing
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Streamlit Inputs
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage (km/l)', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power (BHP)', 0, 200)
seats = st.slider('Number of Seats', 2, 10)

# Prediction Logic
if st.button("Predict"):
    input_data = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])

    # Encoding
    input_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
                                [1,2,3,4,5], inplace=True)
    input_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1,2,3,4], inplace=True)
    input_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1,2,3], inplace=True)
    input_data['transmission'].replace(['Manual', 'Automatic'], [1,2], inplace=True)
    input_data['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                               list(range(1, 32)), inplace=True)

    try:
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Car Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
