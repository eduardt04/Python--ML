import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

from keras.layers import Dropout, Dense
from keras.models import Sequential

from spacy.lang.ro import Romanian

nlp = Romanian()
start_time = time.time()
DELETE_CHARACTERS = re.compile('[.)?":($,%;”0-9]')


def clean_text(text):
    text = text.lower()
    text = DELETE_CHARACTERS.sub('', text)
    text = " ".join(text.split())
    return text


def read_data(file_path):
    f = open(file_path)
    lines = f.readlines()

    lines_split = [line.split('\t') for line in lines]
    data_ids = [int(line[0]) for line in lines_split]

    relevant_data = []
    data_texts = [clean_text(line[1]) for line in lines_split]

    for line in data_texts:
        doc = nlp(line)
        rel_line = " "
        for token in doc:
            if not token.is_stop and len(token) > 1:
                rel_line = rel_line + " " + str(token)
        relevant_data.append(rel_line)

    return data_ids, relevant_data


def read_labels(file_path):
    data_labels = np.loadtxt(file_path, 'int')
    labels = [line[1] for line in data_labels]
    return np.array(labels)-1


def tfidf_data(train, valid, target, test):
    vectorizer = TfidfVectorizer(max_features=150000)
    train = vectorizer.fit_transform(train).toarray()
    valid = vectorizer.transform(valid).toarray()
    target = vectorizer.transform(target).toarray()
    test = vectorizer.transform(test).toarray()
    return train, valid, target, test


def build_model_dnn_text(shape, nClasses, dropout=0.5):
    model = Sequential()
    node = 256
    num_layers = 2

    model.add(Dense(node, input_dim=shape, activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0, num_layers):
        model.add(Dense(node, input_dim=node, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


_, train_data_texts = read_data('data/train_samples.txt')
train_data_labels = read_labels('data/train_labels.txt')

_, validation_data_texts = read_data('data/validation_source_samples.txt')
validation_data_labels = read_labels('data/validation_source_labels.txt')

_, target_data_texts = read_data('data/validation_target_samples.txt')
target_data_labels = read_labels('data/validation_target_labels.txt')

test_data_ids, test_data_texts = read_data('data/test_samples.txt')

for txt in target_data_texts:
    train_data_texts.append(txt)
train_data_labels = np.append(train_data_labels, target_data_labels)

train_data, validation_data, target_data, test_data = tfidf_data(train_data_texts, validation_data_texts,
                                                                 target_data_texts, test_data_texts)

model_dnn = build_model_dnn_text(train_data.shape[1], 2)
model_dnn.fit(train_data, np.array(train_data_labels),
              validation_data=(validation_data, validation_data_labels),
              epochs=5,
              batch_size=128,
              verbose=2,
              )

predictions = model_dnn.predict(target_data)
predictions = [0 if pred[0] > pred[1] else 1 for pred in predictions]

print('accuracy %s' % accuracy_score(predictions, target_data_labels))
print(np.mean(predictions == target_data_labels))
print(classification_report(target_data_labels, predictions))

predictions = model_dnn.predict(test_data)
predictions = [1 if pred[0] > pred[1] else 2 for pred in predictions]

output = pd.DataFrame({'id': test_data_ids,
                       'label': predictions})
output.to_csv('submission.csv', index=False)

print("Run time:", (time.time() - start_time), "seconds")
