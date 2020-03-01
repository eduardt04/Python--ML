import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm


class knn_classifier:

    def __init__(self, train_data, data_labels):
        self.train_images = train_data
        self.train_labels = data_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        if metric == 'l2':
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))
        elif metric == 'l1':
            distances = np.sum(abs(self.train_images - test_image), axis=1)
        else:
            print(f'Error! Metric {metric} is not defined!')

        sort_indexes = np.argsort(distances)
        sort_indexes = sort_indexes[:num_neighbors]
        # nearest_labels = self.train_labels[sort_indexes]
        nearest_labels = np.array(self.train_labels)[sort_indexes.astype(int)]
        occur_count = np.bincount(nearest_labels)

        return np.argmax(occur_count)

    def classify_images(self, test_data, num_neighbors=3, metric='l2'):
        test_images_count = test_data.shape[0]
        predictions = np.zeros(test_images_count, np.int)

        for i in range(test_images_count):
            predictions[i] = self.classify_image(test_data[i, :], num_neighbors=num_neighbors, metric=metric)

        return predictions


def accuracy_score(y_true, y_pred):
    return np.mean((y_true == y_pred))


train_images = np.loadtxt('train_images.txt')
train_labels = np.loadtxt('train_labels.txt', 'float')
train_labels = [int(val) for val in train_labels]

test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'float')
test_labels = [int(val) for val in test_labels]


classifier = knn_classifier(train_images, train_labels)
predicted_labels = classifier.classify_images(test_images, 3, metric='l2')
accuracy = accuracy_score(test_labels, predicted_labels)

print('Accuracy: ', accuracy)
np.savetxt('predictions.txt', predicted_labels)


nn = [1, 3, 5, 7, 9]
accuracies_l1 = np.zeros(len(nn))
accuracies_l2 = np.zeros(len(nn))

for pos, val in enumerate(nn):
    predicted_labels = classifier.classify_images(test_images, num_neighbors=val, metric='l1')
    accuracies_l1[pos] = accuracy_score(test_labels, predicted_labels)
    predicted_labels = classifier.classify_images(test_images, num_neighbors=val, metric='l2')
    accuracies_l2[pos] = accuracy_score(test_labels, predicted_labels)

np.savetxt('accuracies_l1.txt', accuracies_l1)
np.savetxt('accuracies_l2.txt', accuracies_l2)

plt.plot(nn, accuracies_l1)
plt.plot(nn, accuracies_l2)
plt.gca().legend(('L1', 'L2'))
plt.xlabel('k neighbours')
plt.ylabel('accuracy')
plt.show()

