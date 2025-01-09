#easiest way to encode data is bag of words (not effective cause doesn`t keep the order of words)
vocab = {}  # maps word to integer representing it
#You don't need global for modifying the contents of a global mutable object like a dictionary
word_encoding = 1


def bag_of_words(text):
    global word_encoding #to maintain a persistent vocab and word_encoding across multiple calls to bag_of_words

    #You only need global if you reassign the variable itself inside the function.

    words = text.lower().split(" ")  # create a list of all of the words in the text, well assume there is no grammar in our text for this example
    bag = {}  # stores all of the encodings and their frequency

    for word in words:
        if word in vocab:
            encoding = vocab[word]  # get encoding from vocab
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1

        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1

    return bag


text = "this is a test to see if this test will work is is test a a"
bag = bag_of_words(text)
print(bag)
print(vocab)

#-------------------------------------------------------------
#sentiment analysis using lstm(long short-term memory: adds a way to access inputs from any timestep in the past)

from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)
results = model.evaluate(test_data, test_labels)
print(results)