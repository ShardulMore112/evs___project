import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Dictionary containing SO2 values for each city
city_data = {
    'Akola': [13, 15, 14, 14, 14, 16],
    'Ambernath': [25, 27, 18, 20, 21, 19],
    'Amravati': [15, 14, 14, 13, 13, 10],
    'Aurangabad': [14, 14, 19, 17, 15, 16],
    'Badlapur': [24, 27, 18, 21, 22, 22],
    'Bhiwandi': [32, 32, 29, 29, 29, 30],
    'Chandrapur': [4, 4, 4, 9, 14, 14],
    'Dombivali': [27, 27, 21, 29, 21, 23],
    'Jalgaon': [13, 12, 11, 12, 12, 17],
    'Jalna': [11, 10, 9, 10, 10, 15],
    'Kolhapur': [27, 22, 24, 17, 14, 15],
    'Latur': [6, 5, 10, 7, 8, 9],
    'Mumbai': [2, 2, 2, 14, 16, 17],
    'Nagpur': [13, 14, 10, 10, 8, 14],
    'Nashik': [12, 10, 6, 4, 5, 7],
    'Navi Mumbai': [20, 18, 14, 15, 21, 19],
    'Pimpri Chinchwad': [39, 28, 16, 21, 17, 20],
    'Pune': [39, 46, 16, 26, 47, 28],
    'Sangli': [11, 10, 9, 9, 9, 12],
    'Solapur': [16, 17, 19, 11, 18, 15],
    'Thane': [18, 21, 21, 11, 21, 19],
    'Ulhas Nagar': [25, 27, 19, 21, 20, 18],
}

# Define the years
years = np.array([2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)

# Train and save a model for each city
for city, values in city_data.items():
    # Create and train the model
    model = LinearRegression()
    model.fit(years, np.array(values))
    
    # Save the model for the city
    with open(f'{city}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print(f"Model for {city} saved successfully.")
