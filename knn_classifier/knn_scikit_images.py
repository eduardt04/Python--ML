import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


train_images = np.loadtxt('train_images.txt')
train_labels = np.loadtxt('train_labels.txt', 'float')
train_labels = [int(val) for val in train_labels]

test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'float')
test_labels = [int(val) for val in test_labels]


knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn.fit(train_images, train_labels)
predictions = knn.predict(test_images)
print("Accuracy:", np.mean(predictions == test_labels))

nn = [1, 3, 5, 7, 9]
accuracies_euclidean = np.zeros(len(nn))
accuracies_manhattan = np.zeros(len(nn))

for pos, val in enumerate(nn):

    knn_euclidean = KNeighborsClassifier(n_neighbors=val, metric="euclidean")
    knn_euclidean.fit(train_images, train_labels)
    predictions = knn_euclidean.predict(test_images)
    accuracies_euclidean[pos] = np.mean(predictions == test_labels)

    knn_manhattan = KNeighborsClassifier(n_neighbors=val, metric="manhattan")
    knn_manhattan.fit(train_images, train_labels)
    predictions = knn_manhattan.predict(test_images)
    accuracies_manhattan[pos] = np.mean(predictions == test_labels)

np.savetxt('accuracies_euclidean.txt', accuracies_euclidean)
np.savetxt('accuracies_manhattan.txt', accuracies_manhattan)

plt.plot(nn, accuracies_euclidean)
plt.plot(nn, accuracies_manhattan)
plt.gca().legend(('Euclidean', 'Manhattan'))
plt.xlabel('k neighbours')
plt.ylabel('accuracy')
plt.show()
