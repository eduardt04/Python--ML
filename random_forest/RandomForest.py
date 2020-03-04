import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Read data
data_file_path = "data/train.csv"
data = pd.read_csv(data_file_path)

# Create X and y
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]
y = data.SalePrice

# Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Initialize DecisionTree and fit it
dtree_model = DecisionTreeRegressor(random_state=1)
dtree_model.fit(train_X, train_y)

# Make predictions - basic DTree
val_predictions = dtree_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
mln_dtree_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
mln_dtree_model.fit(train_X, train_y)
val_predictions = mln_dtree_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Using random forest
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


