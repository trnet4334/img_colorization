from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.datasets import cifar10
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import rmsprop
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, shutil
import theano
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from skimage.color import rgb2lab
import skimage.color as color
import cv2
from scipy.misc import imresize

# Cnn model parameters
epoch = 10
batch_size = 64
num_classes = 10
validation_split = 0.1

# rgb to gray scale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

''' Start Here '''

# Load Data and resize to 224x224
(x_train32, y_train32), (x_test32, y_test32) = cifar10.load_data()
x_train = np.zeros((5000, 224, 224, 3))
x_trainGray = np.zeros((5000, 224, 224, 3))
x_test = np.zeros((1000, 224, 224, 3))
x_testGray = np.zeros((1000, 224, 224, 3))

for i in range(5000):
    im = x_train32[i,:,:,:]
    largeImg = imresize(im, (224,224,3))
    x_train[i,:,:,:] = largeImg

y_train = np_utils.to_categorical(y_train32[0:5000,:], num_classes)
y_test = np_utils.to_categorical(y_test32[0:1000,:], num_classes)

for i in range(5000):
    im = x_train[i,:,:,:]
    x_trainGray[i,:,:,0] = rgb2gray(im).astype(uint8)
    x_trainGray[i,:,:,1] = x_trainGray[i,:,:,0]
    x_trainGray[i,:,:,2] = x_trainGray[i,:,:,0]

for i in range(1000):
    im = x_test32[i,:,:,:]
    largeImg = imresize(im, (224,224,3))
    x_test[i,:,:,:] = largeImg

for i in range(1000):
    im = x_test[i,:,:,:]
    x_testGray[i,:,:,0] = rgb2gray(im).astype(uint8)
    x_testGray[i,:,:,1] = x_testGray[i,:,:,0]
    x_testGray[i,:,:,2] = x_testGray[i,:,:,0]

x_train[:,:,:,0] -= 103.939
x_train[:,:,:,1] -= 116.779
x_train[:,:,:,2] -= 123.68 
x_test[:,:,:,0] -= 103.939
x_test[:,:,:,1] -= 116.779
x_test[:,:,:,2] -= 123.68 
x_trainGray[:,:,:,0] -= 103.939*0.229+116.779*0.587+123.68*0.114
x_trainGray[:,:,:,1] -= 103.939*0.229+116.779*0.587+123.68*0.114
x_trainGray[:,:,:,2] -= 103.939*0.229+116.779*0.587+123.68*0.114
x_testGray[:,:,:,0] -= 103.939*0.229+116.779*0.587+123.68*0.114
x_testGray[:,:,:,1] -= 103.939*0.229+116.779*0.587+123.68*0.114
x_testGray[:,:,:,2] -= 103.939*0.229+116.779*0.587+123.68*0.114

initial_model = VGG16(weights='imagenet', include_top=True)
x = Dense(10, activation='softmax', name='predictions')(initial_model.layers[-2].output)
model = Model(input=initial_model.input, output=x)
opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch, verbose=1, validation_split=0.1, shuffle=True)
RGBonRGBModelScore = model.evaluate(x_test, y_test, batch_size=64)
grayOnRGBModelScore = model.evaluate(x_testGray, y_test, batch_size=64)

print "loss and accuracy for rgb finetune model"
print RGBonRGBModelScore
print "loss and accuracy for gray scale images testing on the rgb finetune model"
print grayOnRGBModelScore


