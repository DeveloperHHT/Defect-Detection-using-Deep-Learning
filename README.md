# Defect-Detection-using-Deep-Learning
I have developed a deep learning algorythm that uses the Tensorflow library to train the data and create a CNN model. By using this trained model, the program will be able to accurately detect defective products and provide the necessary visual representation. Also basic structure that I use can be applied to almost on anything


-To run this project correctly, first we need to import libraries. "pip" command can be used to instal all of them one by one or use Pycharm to auto install all of them

# IMPORTANT: There are 2 different type of defects on buttons : Dots and Scratches. I will be trying to test my data if oven buttons are defected or not

## -All imports:

import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report


## -Main library that I am working on is Tensorflow. But Matplotlib is also necessary for visual content. Also Numpy and Pandas will be necessary for this project.

### 1-)After importing libraries Create 2 files named "train" and "test". Then create "def_front" and "ok_front" inside both of them.

### 2-)It is better to set data(images or photos) 80% train and 20% test. Test images must be different from train images. More images, better model.

### 3-)Go to the code and change the "folderdirec tory". Must be valid directory of the project folder.

### 4-)You can also customize epoch count and model name that will be created.

### 5-)After all of this code customization run your code.

### 6-)Program will show show something like this:

![Figure_1](https://user-images.githubusercontent.com/73137439/190092534-c0ef3dde-52bd-492d-91b0-d7b277829cc6.png)
### 7-)You can see your data seperated as defected and okay. After Closing that program will run and create a model. It can take time if you have increased number of epochs.

### 8-)After running all the epochs a ".hdf5" file will be created in the project directory. That file is our trained CNN model

### 9-)When the program executes all the instructions it will give some graphs as output like learning curve, distribution. For example:

![Figure_2](https://user-images.githubusercontent.com/73137439/190095216-d7e6126c-d0d9-4c57-8cb3-e7928c24736d.png)

![Figure_3](https://user-images.githubusercontent.com/73137439/190095265-be39b46a-2d65-4f5c-b06a-97209c107f0b.png)

![Figure_4](https://user-images.githubusercontent.com/73137439/190095293-ef530639-3bee-422b-a619-0fd0651443f0.png)

## This curve is not the curve we want. I Just wanted to show an example. The reason behind this is my data. I got 92% accuracy and almost 0 val_los when I have tried images with same light conditions(with 120 images). But as you can see, my data includes different light conditions and less image. So if you want to fix your graph, basicly you can try this steps:


### -Try to use better photos(get cleat and unblurred shots) under similar light conditions
### -Try to increase number of images 
### -Try to increase number of epochs
