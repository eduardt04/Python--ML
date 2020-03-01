# Import LabelEncoder
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# Assigning features and label variables
# First Feature
weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny',
           'Overcast', 'Overcast', 'Rainy']

# Second Feature
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']

# Label or target variable
play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
weather_encoded = le.fit_transform(weather)

temp_encoded = le.fit_transform(temp)
label = le.fit_transform(play)

# Combining weather and temp into single list of tuples
features = list(zip(weather_encoded, temp_encoded))

model = KNeighborsClassifier(n_neighbors=3)

model.fit(features, label)

predicted = model.predict([[0, 2]])  # 0:Overcast, 2:Mild
print(predicted)



