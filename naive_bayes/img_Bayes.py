import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

train_images = np.loadtxt('train_images.txt')
train_labels = np.loadtxt('train_labels.txt', 'float')
test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'float')

train_labels = [int(value) for value in train_labels]
test_labels = [int(value) for value in test_labels]


def values_to_bins(train_data, interval_data):
    train_data = np.digitize(train_data, interval_data)
    return train_data - 1


interval_divisions = [3, 5, 7, 9, 11]

for interval_count in interval_divisions:
    number_of_intervals = interval_count

    intervals = np.linspace(0, 255, number_of_intervals)
    naive_bayes_model = MultinomialNB()

    discreet_train_images = values_to_bins(train_images, intervals)
    naive_bayes_model.fit(discreet_train_images, train_labels)

    discreet_test_images = values_to_bins(test_images, intervals)
    naive_bayes_model.predict(discreet_test_images)

    accuracy = naive_bayes_model.score(discreet_test_images, test_labels)
    print(accuracy)

num_bins = 5

intervals = np.linspace(0, 255, num_bins)
discreet_train_images = values_to_bins(train_images, intervals)
discreet_test_images = values_to_bins(test_images, intervals)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(discreet_train_images, train_labels)
predictions = naive_bayes_model.predict(discreet_test_images)

count = 0

for pos, val in enumerate(test_labels):
    if predictions[pos] != val and count < 4:
        count += 1
        image = train_images[pos, :]
        image = np.reshape(image, (28, 28))
        plt.title(f"Image labeles as: {predictions[pos]} ")
        plt.imshow(image.astype(np.uint8), cmap='gray')
        plt.show()


def confusion_matrix(y_true, y_pred):
    conf_mat = np.zeros((10, 10))
    for position, value in enumerate(y_pred):
        conf_mat[y_true[position]][value] += 1
    return conf_mat


print(confusion_matrix(test_labels, predictions))

