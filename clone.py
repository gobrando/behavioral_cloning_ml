import csv
from scipy import ndimage
import numpy as np
import cv2

lines = []
with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    print(csvfile)
    reader = csv.reader(csvfile)
    header = next(reader)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/'
        images.append(current_path)
    #source_path = line[0]
    #filename = source_path.split('\\')[-1]
    #print('/home/workspace/CarND-Behavioral-Cloning-P3/data/' + filename)
    #center_path = '/home/workspace/CarND-Behavioral-Cloning-P3/data/' + filename
    #center_image = ndimage.imread(center_path)
    #img_flipped = cv2.flip(center_image, 1)
    #images.append(img_flipped)
    
    #img_left = np.asarray(ndimage.imread(current_path + line[1]))
    #img_right = np.asarray(ndimage.imread(current_path + line[2]))
    
    steering_center = float(line[3])
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    #steering_flipped = -steering_center
   
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)
    #measurements.append(steering_flipped)
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')