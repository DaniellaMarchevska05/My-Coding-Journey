import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#  LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Let's look at a one image
IMG_INDEX = 7  # change this to look at other images

plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
#plt.show()


#Each Layer Learns Increasingly Complex Features: First Layer: Learns simple patterns, such as edges, lines, and textures, Second Layer: Combines the simple patterns from the first layer to detect slightly more complex structures, like shapes or parts of objects, Third Layer: Combines these parts into high-level patterns, such as entire objects (e.g., a car, bird, or airplane)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#(3, 3) refers to the size of the convolutional filter (kernel) (means the filter slides over a 3×3 region of the image at a time)
#The numbers 32, 64 specify the number of feature maps in each convolutional layer: First Layer (32 filters): Detects basic features like edges and textures, (64 filters): Detects more complex patterns by combining the basic features
#input shape`s height, width and number of color channels
#pooling reduces the size of feature maps while retaining the most important information and 2, 2 means pooling operation takes the maximum value from each 2×2 region of the feature map

#model.summary()  # let's have a look at our model so far

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print(test_acc)

#--------------------------------------------------------------------------
#so the accuracy was pretty small since we don`t have enough training data, but we can increase it by data augmentation
#This is simply performing random transofrmations on our images so that our model can generalize better. These transformations can be things like compressions, rotations, stretches and even color changes.
# creates a data generator object that transforms images
datagen = ImageDataGenerator(  #ImageDataGenerator from Keras for data augmentation
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

# pick an image to transform
test_img = train_images[20]
img = image.img_to_array(test_img)  # convert image to numpy array
img = img.reshape((1,) + img.shape)  # The (1,) adds a "batch dimension," meaning this is now a batch containing 1 image.The shape of the image changes from (32, 32, 3) to (1, 32, 32, 3)
#why do we need batches? When training a neural network, the data is typically split into smaller chunks called batches rather than processing the entire dataset at once. Instead of augmenting one image at a time, you can augment several images in parallel in a batch, which is faster.

i = 0

#datagen.flow(img) Applies random transformations from datagen to the input image (img)
for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:  # show 4 images and create 4 images out of one
        break

plt.show()
