# -*- coding: utf-8 -*-
!unzip '/content/DATASET.zip'

"""IMPORT"""

import cv2

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

"""DATA

"""


dir_path = '/content/DATASET/CNN letter Dataset/'
list_labels = np.array(os.listdir(dir_path))
imgs = []
labels = []
for label in list_labels:
  img_list = os.listdir(dir_path + label)
  for img in img_list:
    img = os.path.join(dir_path + label + '/' + img)
    imgs.append(img)
    labels.append(label)
imgs = np.array(imgs)
labels = np.array(labels)
print(list_labels)

df = pd.DataFrame({'Path': imgs, 'Labels': labels})
print(df)

"""VIEW DATA"""

def plot_sample(df):
  plt.figure(figsize=(14,10))
  for i in range(20):
      random = np.random.randint(0,len(df))
      plt.subplot(4,5,i+1)
      plt.imshow(cv.imread(df.loc[random, 'Path']))
      plt.title(df.loc[random, 'Labels'], size = 10, color = "black")
      plt.xticks([])
      plt.yticks([])

  plt.show()

plot_sample(df)

"""SPLIT DATASET INTO TRAIN, TEST, VALIDATION"""

Train_df, Test_df = train_test_split(df, test_size = 0.2, random_state = 101)
Train_df, Valid_df = train_test_split(Train_df, test_size = 0.1, random_state = 101)

Train_df = Train_df.reset_index(drop = True)
Test_df = Test_df.reset_index(drop = True)
Valid_df = Valid_df.reset_index(drop = True)

plot_sample(Train_df)

"""GENERATOR"""

dataGen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)

Train_gen = dataGen.flow_from_dataframe(
    Train_df,
    x_col = 'Path',
    y_col = 'Labels',
    target_size = (128, 128),
    batch_size = 512,
    class_mode = 'categorical',
    color_mode='rgb'
)

Test_gen = dataGen.flow_from_dataframe(
    Test_df,
    x_col = 'Path',
    y_col = 'Labels',
    target_size = (128, 128),
    batch_size = 512,
    class_mode = 'categorical',
    color_mode='rgb'
)

Valid_gen = dataGen.flow_from_dataframe(
    Valid_df,
    x_col = 'Path',
    y_col = 'Labels',
    target_size = (128, 128),
    batch_size = 512,
    class_mode = 'categorical',
    color_mode='rgb'
)

imgs,labels=next(Train_gen) # get a sample batch from the generator
plt.figure(figsize=(14, 10))
for i in range(20):
      random = np.random.randint(0,len(imgs))
      plt.subplot(4,5,i+1)
      plt.imshow(imgs[random])
      plt.title(list_labels[np.argmax(labels[random])], size = 10, color = "black")
      plt.xticks([])
      plt.yticks([])
plt.show()

"""CREATE MODEL"""

model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten
    tf.keras.layers.Flatten(),

    # Fully connected + Dropout => prevent overfitting
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # Output
    tf.keras.layers.Dense(len(list_labels), activation='softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

"""TRAIN MODEL"""

model.fit(Train_gen, epochs = 10, validation_data = Valid_gen, verbose = 1)

test_loss, test_acc = model.evaluate(Test_gen, verbose=2)

print("Test accuracy: ", test_acc)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/DATA
model.save('Model_classification.h5')

imgs, labels = next(Test_gen)

imgs[0].shape

prediction = model.predict(imgs)

plt.figure(figsize=(14, 10))
for i in range(20):
      random = np.random.randint(0,len(imgs))
      plt.subplot(4,5,i+1)
      plt.imshow(imgs[random])
      plt.title('Prediction value: ' + list_labels[np.argmax(prediction[random])], size = 10, color = "black")
      plt.xticks([])
      plt.yticks([])
plt.show()

"""CHECK MODEL IN NEW IMAGE"""

list_labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

model = tf.keras.models.load_model('/content/Model_classification.h5')

from tensorflow.python.ops.gen_logging_ops import image_summary
def load_image(filename):
  image = np.array(tf.keras.preprocessing.image.load_img(filename, target_size = (128, 128), color_mode = 'rgb'))
  # _,image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
  image = tf.keras.preprocessing.image.img_to_array(image)
  image = image.astype('float32')
  image = image / 255.0
  #image = np.array([image])
  return image

def show_predict(img, model):
  plt.figure()
  plt.imshow(img)
  plt.title('Prediction value: ' + list_labels[np.argmax(model.predict(np.array([img])))], size = 10, color = "black")
  plt.xticks([])
  plt.yticks([])
  plt.show()

img = load_image('/content/gdrive/MyDrive/DATA/TEST_IMG/kisspng-letter-alphabet-green-image-a-letter-png-photo-png-all-5c7ed229efeba3.6859165815518152099827.jpg')
show_predict(img, model)

img = load_image('/content/gdrive/MyDrive/DATA/TEST_IMG/y-nghia-cua-so-5-theo-kinh-dich-626454.jpg')
show_predict(img, model)

img = load_image('/content/gdrive/MyDrive/DATA/TEST_IMG/maxresdefault.jpg')
show_predict(img, model)

img = load_image('/content/gdrive/MyDrive/DATA/TEST_IMG/Untitled.png')
show_predict(img, model)

