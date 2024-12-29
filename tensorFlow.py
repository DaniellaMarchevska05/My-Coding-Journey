from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

rank1_tensor = tf.Variable(["Test"], tf.string)
rank2_tensor = tf.Variable([[[["test", "ok"], ["test","yes"]]]], tf.string)

tf.rank(rank2_tensor)
#print(rank2_tensor.shape) #output: (1, 1, 2, 2) - in 1st dimension 1 elem, in 2nd dimension 1 el, in 3rd dimension 2 el, in 4 dimension 2 el

#!the size of original and reshaped tensors has to be the same! (just mult: 1*2*3 = 2*3*1)
tensor1 = tf.ones([1, 2, 3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2, 3, 1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
# this will reshape the tensor to [3,3]

#print(tensor1)
# print(tensor2)
# print(tensor3)

# Creating a 2D tensor
matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32)
# print(tf.rank(tensor))
# print(tensor.shape)


# Now lets select some different rows and columns from our tensor

three = tensor[0,2]  # selects the 3rd element from the 1st row
# print(three)  # -> 3

row1 = tensor[0]  # selects the first row
# print(row1)

column1 = tensor[:, 0]  # selects the first column
# print(column1)


#sequence[start:stop:step]
row_2_and_4 = tensor[1::2]  # selects second and fourth row
# print(row_2_and_4)

column_1_in_row_2_and_3 = tensor[1:3, 0]
# print(column_1_in_row_2_and_3)


#----------------------------------------------------------
#linear regression with keras
# Load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')    # evaluation data
y_train = dftrain.pop('survived')  # target variable for training
y_eval = dfeval.pop('survived')    # target variable for evaluation

# Preprocessing: Feature columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

#Converts categorical data into numerical format using pandas.Categorical
def one_hot_encode(df, columns):
    for col in columns:
        df[col] = pd.Categorical(df[col])
        df[col] = df[col].cat.codes
    return df

dftrain = one_hot_encode(dftrain, CATEGORICAL_COLUMNS)
dfeval = one_hot_encode(dfeval, CATEGORICAL_COLUMNS)

# Fill missing values
dftrain.fillna(dftrain.mean(), inplace=True)
dfeval.fillna(dfeval.mean(), inplace=True)

# Normalize numeric columns
for col in NUMERIC_COLUMNS:
    dftrain[col] = (dftrain[col] - dftrain[col].mean()) / dftrain[col].std()
    dfeval[col] = (dfeval[col] - dfeval[col].mean()) / dfeval[col].std()

# Convert to numpy arrays
X_train = dftrain.to_numpy()**
X_eval = dfeval.to_numpy()

# Build the Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_eval, y_eval)
print(f"Evaluation accuracy: {accuracy:.2f}")

# Predict probabilities
pred_probs = model.predict(X_eval).flatten()

# Plot predicted probabilities
plt.hist(pred_probs, bins=20, color='pink', alpha=0.7)
plt.title('Predicted Probabilities')
plt.xlabel('Probability of Survival')
plt.ylabel('Frequency')
plt.show()
