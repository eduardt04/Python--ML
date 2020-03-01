import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

train_images = np.loadtxt('train_images.txt', unpack=True)
train_labels = np.loadtxt('train_labels.txt', 'float')
test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'float')

train_labels = [int(value) for value in train_labels]
test_labels = [int(value) for value in test_labels]