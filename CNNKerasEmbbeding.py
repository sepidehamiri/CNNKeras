"""This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
"""

from __future__ import print_function

import csv
import os

import numpy
import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR = '/home/mos/Desktop/CNNKeras'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 2000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
TRAIN_SPLIT = 0.6
TEST_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')
data = []
texts = []  # list of text samples

labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
with open('data.csv', 'r') as f:
    read = csv.reader(f)
    next(read)
    for row in read:
        texts.append(row[0])
        labels.append(row[1])
print('Found %s texts.' % len(texts))
MultiLabelBinarizer().fit_transform(labels)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
list_of_tuple_labels = []
for label in labels:
    list_of_tuple_labels.append(tuple(label.split(' ')))
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(list_of_tuple_labels)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# Converts a class vector (integers) to binary class matrix.
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * len(data))
num_test_samples = int(TEST_SPLIT * len(data))

data = numpy.array(data)  # convert array to numpy type array

x_val = data[:-num_validation_samples]
y_val = labels[:-num_validation_samples]
x_test = data[num_validation_samples:-num_test_samples]
y_test = labels[num_validation_samples:-num_test_samples]
x_train = data[-num_test_samples:]
y_train = labels[-num_test_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
print(x_val)
print(y_val)
print(type(x_val))
print(type(y_val))
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels[0]), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=1,
          validation_data=(x_val, y_val))

y_predict = model.predict(x_test)
for j in range(0, len(y_predict)):
    y_predict[j] = [1 if x >= 0.5 else 0 for x in y_predict[j]]

scores = model.evaluate(x_test, y_test)
print('\nTest loss:', scores[0])
print('Test accuracy:', scores[1])


def precision(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1


def recall(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    return len(i) / len(y_true)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)


f1_score_total = 0
for i in range(0, len(y_test)):
    f1_score_total += f1(y_test, y_predict)

f1_score_mean = f1_score_total / len(y_test)
