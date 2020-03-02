from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

data_file_path = 'data/train.csv'
data = pd.read_csv(data_file_path)

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[feature_names]
y = data.SalePrice

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


def get_mae(max_leaf_nodes, data_train_X, data_val_X, data_train_y, data_val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(data_train_X, data_train_y)
    preds_val = model.predict(data_val_X)
    mae = mean_absolute_error(data_val_y, preds_val)
    return mae


for num_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(num_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(num_leaf_nodes, my_mae))


scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in [5, 25, 50, 100, 250, 500]}
best_tree_size = min(scores, key=scores.get)
print(best_tree_size)

