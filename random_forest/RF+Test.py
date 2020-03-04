import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read data
train_data_path = "data/train.csv"
# path to file with data for  predictions
test_data_path = "data/test.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Create X and y
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features]
y = train_data.SalePrice
test_X = test_data[features]

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X, y)

test_predictions = rf_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})
output.to_csv('submission.csv', index=False)

results = pd.read_csv("submission.csv")
print(results.head())
