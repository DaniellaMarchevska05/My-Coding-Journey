from __future__ import absolute_import, division, print_function, unicode_literals
#paste it when working with tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set to '1' for warnings, '2' for errors, '3' for critical

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
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

#----------------------------------------------------
# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')# cutting column survived from dftrain and creating y_train with just this column
y_eval = dfeval.pop('survived')

# print("\n", dftrain.head())#first five rows
# print(y_train) #1 stands for surveved, 0 - not

# print(dftrain.describe()) #some statistical analysis of our data (f.e. count, mean, std, min, max, 25%, 75%)

# print(dftrain.shape) #627 rows/entries, 9 columns/featuresdftrain.age.hist(bins=20)

#dftrain.age.hist(bins=20)

#dftrain.sex.value_counts().plot(kind='barh')

#dftrain['class'].value_counts().plot(kind='barh')

pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()