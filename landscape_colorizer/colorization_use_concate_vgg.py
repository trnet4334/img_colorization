from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Merge
from keras.layers.convolutional import UpSampling2D, ZeroPadding2D, Convolution2D, MaxPooling2D,Conv2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16

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
img_rows, img_cols = 96, 96   # Image dimensions after resizing
bin_num = 20                     # For classification : Since a and b channel contains continous value from -100 to 100, we bin them to several classes
input_channels = 3               # The paper use 3 duplicated channel as input since pre-trained network has 3 channel, but we can use 1 if we are not using VGG-16
test_img_num = 60                 # Use first-n files in the data folder to test the model
lab_channels = ['l', 'a', 'b']

# Cnn model parameters
era = 1000
epoch = 3
batch_size = 100
validation_split = 0.1

# Paths
img_input_path = "./combined/"
img_output_path = "./predict_output_concat_vgg/"
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

X = array(X).astype(np.float32)
Y = array(Y)
X_l = array(X_l)

X = X - 45.0

print X.shape
print Y.shape

l_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

l_model.summary()

 # Conv-Pooling Layers
model1 = Sequential()
model1.add(ZeroPadding2D((1,1), name='m1_c2', input_shape=(img_rows, img_cols, input_channels)))
model1.add(Convolution2D(64, 3, 3, input_shape=(img_rows, img_cols, input_channels), name='m1_c3', activation='relu'))
#model1.add(ZeroPadding2D((1,1), name='m1_c4', ))
model1.add(Convolution2D(64, 3, 3, border_mode='same', name='m1_c5', activation='relu'))
model1.add(MaxPooling2D((2,2), name='m1_c6', strides=(2,2)))

#model1.add(ZeroPadding2D((1,1), name='m1_c7'))
model1.add(Convolution2D(128, 3, 3, border_mode='same', name='m1_c8', activation='relu'))
#model1.add(ZeroPadding2D((1,1), name='m1_c9'))
model1.add(Convolution2D(128, 3, 3, border_mode='same', name='m1_c10', activation='relu'))
model1.add(MaxPooling2D((2,2), name='m1_c11', strides=(2,2)))

#model1.add(ZeroPadding2D((1,1), name='m1_c12'))
model1.add(Convolution2D(256, 3, 3, border_mode='same', name='m1_c13', activation='relu'))
#model1.add(ZeroPadding2D((1,1), name='m1_c14'))
model1.add(Convolution2D(256, 3, 3, border_mode='same', name='m1_c15', activation='relu'))
#model1.add(ZeroPadding2D((1,1), name='m1_c16'))
model1.add(Convolution2D(256, 3, 3, border_mode='same', name='m1_c17', activation='relu'))
model1.add(MaxPooling2D((2,2), name='m1_c18', strides=(2,2)))

#model1.add(ZeroPadding2D((1,1), name='m1_c19'))
model1.add(Convolution2D(512, 3, 3, border_mode='same', name='m1_c20', activation='relu'))
#model1.add(ZeroPadding2D((1,1), name='m1_c21'))
model1.add(Convolution2D(512, 3, 3, border_mode='same', name='m1_c22', activation='relu'))
#model1.add(ZeroPadding2D((1,1), name='m1_c23'))
model1.add(Convolution2D(512, 3, 3, border_mode='same', name='m1_c24', activation='relu'))
model1.add(MaxPooling2D((2,2), name='m1_c25', strides=(2,2)))
for i in xrange(0, len(model1.layers), 1):
        model1.layers[i].set_weights(l_model.layers[i].get_weights())
        model1.layers[i].trainable = False

model1.add(Convolution2D(256, 1, 1, name='m1_c31', activation='relu'))
model1.add(BatchNormalization(name='m1_c32'))
model1.add(UpSampling2D((2,2), name='m1_c33'))
model1.summary()


model2 = Sequential()
model2.add(ZeroPadding2D((1,1), name='m2_c2', input_shape=(img_rows, img_cols, input_channels)))
model2.add(Convolution2D(64, 3, 3, input_shape=(img_rows, img_cols, input_channels), name='m2_c3', activation='relu'))
#model2.add(ZeroPadding2D((1,1), name='m2_c4', ))
model2.add(Convolution2D(64, 3, 3, border_mode='same', name='m2_c5', activation='relu'))
model2.add(MaxPooling2D((2,2), name='m2_c6', strides=(2,2)))

#model2.add(ZeroPadding2D((1,1), name='m2_c7'))
model2.add(Convolution2D(128, 3, 3, border_mode='same', name='m2_c8', activation='relu'))
#model2.add(ZeroPadding2D((1,1), name='m2_c9'))
model2.add(Convolution2D(128, 3, 3, border_mode='same', name='m2_c10', activation='relu'))
model2.add(MaxPooling2D((2,2), name='m2_c11', strides=(2,2)))

#model2.add(ZeroPadding2D((1,1), name='m2_c12'))
model2.add(Convolution2D(256, 3, 3, border_mode='same', name='m2_c13', activation='relu'))
#model2.add(ZeroPadding2D((1,1), name='m2_c14'))
model2.add(Convolution2D(256, 3, 3, border_mode='same', name='m2_c15', activation='relu'))
#model2.add(ZeroPadding2D((1,1), name='m2_c16'))
model2.add(Convolution2D(256, 3, 3, border_mode='same', name='m2_c17', activation='relu'))
model2.add(MaxPooling2D((2,2), name='m2_c18', strides=(2,2)))

for i in xrange(0, len(model2.layers), 1):
        model2.layers[i].set_weights(l_model.layers[i].get_weights())
        model2.layers[i].trainable = False

model2.add(BatchNormalization(name='m2_c32'))
model2.summary()


model3 = Sequential()
model3.add(ZeroPadding2D((1,1), name='m3_c2', input_shape=(img_rows, img_cols, input_channels)))
model3.add(Convolution2D(64, 3, 3, input_shape=(img_rows, img_cols, input_channels), name='m3_c3', activation='relu'))
#model3.add(ZeroPadding2D((1,1), name='m3_c4', ))
model3.add(Convolution2D(64, 3, 3, border_mode='same', name='m3_c5', activation='relu'))
model3.add(MaxPooling2D((2,2), name='m3_c6', strides=(2,2)))

#model3.add(ZeroPadding2D((1,1), name='m3_c7'))
model3.add(Convolution2D(128, 3, 3, border_mode='same', name='m3_c8', activation='relu'))
#model3.add(ZeroPadding2D((1,1), name='m3_c9'))
model3.add(Convolution2D(128, 3, 3, border_mode='same', name='m3_c10', activation='relu'))
model3.add(MaxPooling2D((2,2), name='m3_c11', strides=(2,2)))

for i in xrange(0, len(model3.layers), 1):
        model3.layers[i].set_weights(l_model.layers[i].get_weights())
        model3.layers[i].trainable = False

model3.add(BatchNormalization(name='m3_c32'))
model3.summary()


model4 = Sequential()
model4.add(ZeroPadding2D((1,1), name='m4_c2', input_shape=(img_rows, img_cols, input_channels)))
model4.add(Convolution2D(64, 3, 3, input_shape=(img_rows, img_cols, input_channels), name='m4_c3', activation='relu'))
#model4.add(ZeroPadding2D((1,1), name='m4_c4', ))
model4.add(Convolution2D(64, 3, 3, border_mode='same', name='m4_c5', activation='relu'))
model4.add(MaxPooling2D((2,2), name='m4_c6', strides=(2,2)))

for i in xrange(0, len(model4.layers), 1):
        model4.layers[i].set_weights(l_model.layers[i].get_weights())
        model4.layers[i].trainable = False

model4.add(BatchNormalization(name='m4_c32'))
model4.summary()

print "\n\n\n\n==============================="
model1.summary()
model2.summary()
print "===============================\n\n\n"
model5 = Sequential()
model5.add(Merge([model1, model2], mode = 'concat', name='m5_c0'))
# model5.add(ZeroPadding2D((1,1), name='m5_c1'))
model5.add(Convolution2D(128, 3, 3, border_mode='same', name='m5_c2', activation='relu'))
model5.add(UpSampling2D((2,2), name='m5_c33'))
model5.summary()

model6 = Sequential()
model6.add(Merge([model5, model3], mode = 'concat', name='m6_c0'))
# model6.add(ZeroPadding2D((1,1), name='m6_c1'))
model6.add(Convolution2D(64, 3, 3, border_mode='same', name='m6_c2', activation='relu'))
model6.add(UpSampling2D((2,2), name='m6_c33'))
model6.summary()

print "\n\n\nModel7"
model7 = Sequential()
model7.add(Merge([model6, model4], mode = 'concat', name='m7_c0'))
model7.add(Convolution2D(3, 3, 3, border_mode='same', name='m7_c4', activation='relu'))
model7.add(UpSampling2D((2,2), name='76_c33'))
model7.summary()

print "\n\n\nModel8"
model8 = Sequential()
model8.add(ZeroPadding2D((0,0), name='m8_c2', input_shape=(img_rows, img_cols, input_channels)))
model8.summary()

print "\n\n\nModel9"
model9 = Sequential()
model9.add(Merge([model7, model8], mode = 'concat', name='m9_c0'))
model9.add(Convolution2D(2 * bin_num, 3, 3, border_mode='same', name='m9_c4', activation='relu'))
model9.add(Flatten())
model9.summary()

model9.add(Reshape((img_rows * img_cols * 2, bin_num)))
model9.add(Activation('softmax', name="act"))
model9.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["acc"])
model9.summary()

for j in xrange(era):
	hist = model9.fit([X[test_img_num:], X[test_img_num:], X[test_img_num:], X[test_img_num:], X[test_img_num:]], Y[test_img_num:], batch_size=batch_size, nb_epoch=epoch,
	              verbose=1, validation_split=validation_split, shuffle=True)

	if j % 10 == 0:
		for i in xrange(0, test_img_num):
			xx = X[i].flatten().reshape(1, img_rows, img_cols, input_channels)
			result = model9.predict_classes([xx, xx, xx, xx, xx])
			print result
			reshaped = result.reshape(2, img_rows, img_cols)
			a, b = get_img_ab_unbinned(reshaped[0], reshaped[1])
			im = reconstruct_image_by_lab_channels(X_l[i], a, b)
			im.save(img_output_path + imlist[i] + "_predicted_" + "era_" +str(j) + ".jpg", "jpeg")

	model_json = model9.to_json()
	with open("colorize_with_pretrain.json", "w") as json_file:
	  json_file.write(model_json)
	model9.save_weights("colorize_with_pretrain.hdf5", overwrite=True)
