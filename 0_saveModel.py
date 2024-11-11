import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import warnings

# Ignore warnings about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load the California housing dataset
dataCH = fetch_california_housing()
dfCH = pd.DataFrame(dataCH.data, columns=dataCH['feature_names'])
dfCH['PRICE'] = dataCH.target

# Split dataset: 98% training + 2% testing
x = dfCH[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
          'Population', 'AveOccup', 'Latitude', 'Longitude']]
y = dfCH['PRICE']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.02, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make a prediction with new data
new_data = [[4.240700, 40, 5.694362, 1.032641, 1851, 2.746291, 34.16, -117.99]]
prediction = model.predict(new_data)
print(prediction)

# Save model using joblib
joblib.dump(model, 'modelJoblib')

# Optional: Save model using pickle
# import pickle
# with open('modelPickle.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)
