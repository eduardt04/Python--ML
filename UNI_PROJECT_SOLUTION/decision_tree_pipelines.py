import time, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing

total_lines = 22000
start_time = time.time()
DELETE_CHARACTERS = re.compile('[.)?":($,%;”0-9]')


def clean_text(text):
    text = text.lower()  # lowercase text
    text = DELETE_CHARACTERS.sub('', text)
    text = " ".join(text.split())
    return text


def read_data(file_path):
    f = open(file_path)
    lines = f.readlines()
    lines = lines[:total_lines]
    lines_split = [line.split('\t') for line in lines]
    data_ids = [int(line[0]) for line in lines_split]
    data_texts = [clean_text(line[1]) for line in lines_split]
    return data_ids, data_texts


def read_labels(file_path):
    data_labels = np.loadtxt(file_path, 'int')
    data_labels = data_labels[:total_lines]
    labels = [line[1] for line in data_labels]
    return labels


train_data_ids, train_data_texts = read_data('data/train_samples.txt')
train_data_labels = read_labels('data/train_labels.txt')

nb = Pipeline([('vect', CountVectorizer()),
               #('norm', preprocessing.StandardScaler(with_mean=False)),
               ('tfidf', TfidfTransformer(norm='l2')),
               ('clf', tree.DecisionTreeClassifier()),
              ])

validation_data_ids, validation_data_texts = read_data('data/validation_source_samples.txt')
validation_data_labels = read_labels('data/validation_source_labels.txt')

for id, txt in enumerate(validation_data_texts):
    train_data_texts.append(txt)
    train_data_labels.append(validation_data_labels[id])

target_data_ids, target_data_texts = read_data('data/validation_target_samples.txt')
target_data_labels = read_labels('data/validation_target_labels.txt')

#for id, txt in enumerate(target_data_texts):
#    train_data_texts.append(txt)
#    train_data_labels.append(target_data_labels[id])

nb.fit(train_data_texts, train_data_labels)

predictions = nb.predict(target_data_texts)

print('accuracy %s' % accuracy_score(predictions, target_data_labels))
print(classification_report(target_data_labels, predictions))

test_data_ids, test_data_texts = read_data('data/test_samples.txt')
predictions = nb.predict(test_data_texts)

output = pd.DataFrame({'id': test_data_ids,
                       'label': predictions})
output.to_csv('submission.csv', index=False)

print("Run time:", (time.time() - start_time), "seconds")
