# load ML model : pickle
# import pickle

# with open('modelPickle', 'rb') as modelku:
#     modelLoad = pickle.load(modelku)

import joblib
import pandas as pd
import numpy as np

# Load the saved model
model = joblib.load('modelJoblib')

# Define new data with the same column names used during training
new_data = pd.DataFrame(np.array([[4.240700, 40, 5.694362, 1.032641, 1851, 2.746291, 34.16, -117.99]]),
                        columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                                 'Population', 'AveOccup', 'Latitude', 'Longitude'])

# Make a prediction
prediction = model.predict(new_data)
print(prediction)
