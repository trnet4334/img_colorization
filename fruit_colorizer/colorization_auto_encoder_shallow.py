from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Conv2D, UpSampling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

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

# General parameters
img_rows, img_cols =  96, 96   # Image dimensions after resizing
bin_num = 20                     # For classification : Since a and b channel contains continous value from -100 to 100, we bin them to several classes
input_channels = 1               # The paper use 3 duplicated channel as input since pre-trained network has 3 channel, but we can use 1 if we are not using VGG-16
test_img_num = 40                 # Use first-n files in the data folder to test the model
lab_channels = ['l', 'a', 'b']

# Cnn model parameters
era = 1000
epoch = 3
batch_size = 64
validation_split = 0.1

# Paths
img_input_path = "./fruitdata/"
img_output_path = "./predict_output_auto_encoder_shallow/"
img_reconstructed_path = "./reconstructed_input_after_bining/"
img_channels_path = "./channels_img/"

def save_img_of_channel(img_lab, channel, name="img"):
	img_lab_cp = img_lab.copy()

	# Delete the rest channels by setting them to 0
	if channel == 'l':
		img_lab_cp[:,:,1:] = 0
	elif channel == 'a':
		img_lab_cp[:,:,0] = 0
		img_lab_cp[:,:,2] = 0
	elif channel == 'b':
		img_lab_cp[:,:,:2] = 0
	else:
		print "[ERROR!!] The channel should be 'l', 'a' or 'b' "
		return
	img_rgb_channel = color.lab2rgb(img_lab_cp)
	im = Image.fromarray((img_rgb_channel * 255).astype(uint8))
	im.save(img_channels_path + name + "_" + channel + ".jpg", "jpeg")

def save_image_by_channels(img_lab, name):
	# Seperate the image channels L a* and b*
	for i in xrange(0, len(lab_channels)):
		img = img_lab[:,:,i]
		save_img_of_channel(img_lab, lab_channels[i], name=name)

def reconstruct_image_by_lab_channels(img_l, img_a, img_b):
	img = array([img_l.T, img_a.T, img_b.T]).T
	img_rgb_channel = color.lab2rgb(img)
	im = Image.fromarray((img_rgb_channel * 255).astype(uint8))
	return im

def get_img_ab_binned(img_lab):
	img_a = img_lab[:,:,1]
	img_b = img_lab[:,:,2]

	img_a_binned = ((img_a + 100) * bin_num) / 200
	img_b_binned = ((img_b + 100) * bin_num) / 200

	return img_a_binned.astype(int), img_b_binned.astype(int)

def get_img_ab_unbinned(img_a_binned, img_b_binned):
	img_a_unbinned = ((img_a_binned * 200) / bin_num) - 100.0
	img_b_unbinned = ((img_b_binned * 200) / bin_num) - 100.0

	return img_a_unbinned, img_b_unbinned

def save_input_image_after_bining(img_lab, name='img'):
	# Use this function to test how bin_num affect the original input image
	img_a_binned, img_b_binned = get_img_ab_binned(img_lab)
	img_a_unbinned, img_b_unbinned = get_img_ab_unbinned(img_a_binned, img_b_binned)
	im = reconstruct_image_by_lab_channels(img_lab[:,:,0], img_a_unbinned, img_b_unbinned)
	im.save(img_reconstructed_path + name + "_reconstructed_after_bining.jpg", "jpeg")

def get_duplicated_l_channel(img_l, channels):
	img_l_duplicated = []
	for i in xrange(channels):
		img_l_duplicated.append(img_l.T)
	result = array(img_l_duplicated).T
	return result


''' Start Here '''

imlist = os.listdir(img_input_path)
imlist.sort()

# ''' For playing with lab images and also testing the affect of bining '''
for i in xrange(test_img_num):
	# Save image of each channel (l, a, b)
	img_rgb = array(Image.open(img_input_path + imlist[i]).resize((img_rows,img_cols)))
	img_lab = rgb2lab(img_rgb)
	save_image_by_channels(img_lab, imlist[i])

	# Test the color distortion of input image after bining
	save_input_image_after_bining(img_lab, name = imlist[i])

''' For training and testing cnn model '''

X = []   # Traning inputs
X_l = [] # Keep the l channel to reconstruct the image from lab to rgb
Y = []   # Traning labels

count = 1;
for img in imlist:
	print "loading data .... " + str(count) + "/" +str(len(imlist))
	img_rgb = array(Image.open(img_input_path + img).resize((img_rows,img_cols)))
	img_lab = rgb2lab(img_rgb)
	img_a_binned, img_b_binned = get_img_ab_binned(img_lab)
	img_y = np.append(img_a_binned.flatten(), img_b_binned.flatten())
	y = np_utils.to_categorical(img_y, bin_num)
	X.append(get_duplicated_l_channel(img_lab[:,:,0], input_channels)) # The paper use 3 duplicated l channel as network input
	X_l.append(img_lab[:,:,0])
	Y.append(y)
	count += 1

X = array(X)
Y = array(Y)
X_l = array(X_l)

print X.shape
print Y.shape

# Use the trained model to do prediction


model = Sequential()
model.add(Conv2D(20, 3, input_shape=(img_rows, img_cols, 1), border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(20, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(20, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(40, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(40, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(40, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(40, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(40, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(40, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(20, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(20, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(20, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(40, 1, border_mode='same'))
model.add(Activation('relu'))
model.summary()

model.add(Flatten())
#model.add(Dense(900))
#model.add(Dense(img_rows * img_cols * 2 * bin_num, name='den'))
model.add(Reshape((img_rows * img_cols *2, bin_num)))
model.add(Activation('softmax', name="act"))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["acc"])

model.summary()

for j in xrange(era):
	hist = model.fit(X[test_img_num:], Y[test_img_num:], batch_size=batch_size, nb_epoch=epoch,
	              verbose=1, validation_split=validation_split, shuffle=True)

	if j % 10 == 0:
		for i in xrange(0, test_img_num):
			result = model.predict_classes(X[i].flatten().reshape(1, img_rows, img_cols, input_channels))
			print result
			reshaped = result.reshape(2, img_rows, img_cols)
			a, b = get_img_ab_unbinned(reshaped[0], reshaped[1])
			im = reconstruct_image_by_lab_channels(X_l[i], a, b)
			im.save(img_output_path + imlist[i] + "_predicted_" + "era_" +str(j) + ".jpg", "jpeg")

	model_json = model.to_json()
	with open("colorize_with_pretrain.json", "w") as json_file:
	  json_file.write(model_json)
	model.save_weights("colorize_with_pretrain.hdf5", overwrite=True)
