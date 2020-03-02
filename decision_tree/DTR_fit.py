from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Path of the file to read
data_file_path = 'data/train.csv'

# Load data
data = pd.read_csv(data_file_path)

# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Predict Sale Price
y = data.SalePrice

# Select data corresponding to features in feature_names
X = data[feature_names]

# Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define model
prediction_model = DecisionTreeRegressor(random_state=1)

# Fit model
prediction_model.fit(train_X, train_y)

# Get predicted prices on validation data
val_predictions = prediction_model.predict(val_X)

# Determine MAE of the predictions
print(mean_absolute_error(val_y, val_predictions))
