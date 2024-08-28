import streamlit as st
import pickle
import numpy as np

# Load the models
city_models = {}
city_names = [
    'Akola', 'Ambernath', 'Amravati', 'Aurangabad', 'Badlapur', 
    'Bhiwandi', 'Chandrapur', 'Dombivali', 'Jalgaon', 'Jalna', 
    'Kolhapur', 'Latur', 'Mumbai', 'Nagpur', 'Nashik', 
    'Navi Mumbai', 'Pimpri Chinchwad', 'Pune', 'Sangli', 'Solapur', 
    'Thane', 'Ulhas Nagar'
]

for city in city_names:
    try:
        with open(f'{city}_model.pkl', 'rb') as f:
            city_models[city] = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model for {city} not found.")

# Streamlit app
st.title("SO2 Level Prediction")

# User input
city = st.selectbox("Select a city", city_names)

# Fixed year
year = 2023

if city in city_models:
    model = city_models[city]
    prediction = model.predict(np.array([[year]]))[0]
    st.write(f"Predicted SO2 level for {city} in {year}: {prediction:.2f}")
else:
    st.write("Model not available for the selected city.")
