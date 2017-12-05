
# coding: utf-8



import csv
import cv2
import numpy as np


# ## Preprocess the information of the laps
# We are going to use the information of three laps. 

def path_to_files(f):
    """ 
    Return a list with the path of the image files. 
    Arguments:
    f: path to the folder that contains the images.
    Returns:
    A python list that includes the path to the images.
    """ 
    lines = []
    with open(f) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)        
    return lines 



lines = path_to_files('./data_car_simulator/driving_log.csv')
lines2 = path_to_files('./data_car_simulator2/driving_log.csv')
lines3 = path_to_files('./data_car_simulator3/driving_log.csv')


images = []
measurements = []


correction = 0.2  # this is a parameter to tune 


for line in lines: 
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data_car_simulator/IMG/' + filename 
        image = cv2.imread(current_path)
        images.append(image)
        if i == 0:
           measurement = float(line[3])
        elif i == 1:
           measurement = float(line[3]) + correction
        else:
           measurement = float(line[3]) - correction 
        
        measurements.append(measurement)        


correction = 0.2  # this is a parameter to tune 


for line in lines2: 
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data_car_simulator2/IMG/' + filename 
        image = cv2.imread(current_path)
        images.append(image)
        if i == 0:
           measurement = float(line[3])
        elif i == 1:
           measurement = float(line[3]) + correction
        else:
           measurement = float(line[3]) - correction 
        
        measurements.append(measurement)        


correction = 0.2  # this is a parameter to tune 


for line in lines3: 
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data_car_simulator3/IMG/' + filename 
        image = cv2.imread(current_path)
        images.append(image)
        if i == 0:
           measurement = float(line[3])
        elif i == 1:
           measurement = float(line[3]) + correction
        else:
           measurement = float(line[3]) - correction 
        
        measurements.append(measurement)        


# ## Exploratory Analysis 

import matplotlib 
matplotlib.use('agg') # it is necessary to use matplotlib in the terminal with the Python command.
import matplotlib.pyplot as plt


# the histogram of the data
nbins = 5

n, bins, patches = plt.hist(measurements, nbins, facecolor='g', alpha=0.75)

plt.xlabel('Steering angle')
plt.ylabel('Frequency')
plt.title('Histogram of Steering angles')
plt.show()


# The data is not uniform, that is why the model is specialized in the steering angle localized in the center. In order to fix it, we are going to generate data that will be uniformly distributed from the information of the three laps. 


from random import randint

def select_bin(bins, num_bins):
    """ 
    Return a sample of the bins.
    Arguments:
    bins: the bins 
    num_bins: the number of bins
    Returns:
    A python tuple with a sample of the bins.
    """     
    
    start = randint(0, num_bins-1)
    end = start  + 1 
    return bins[start], bins[end]
    
def select_element_bin(bin, measurements, images):
    """ 
    Return a sample of angle measurements and images.
    Arguments:
    bins: the bin 
    measurement: a python list contains the information of angle measurements.
    images: a python list containts the information of the images.
    Returns:
    A python tuple with a sample of angle measurements and images.
    """        
    
    indexs = np.logical_and(measurements > bin[0], measurements < bin[1])
    indexs = np.where(indexs)
    indexs = indexs[0]
    indexs = np.array(indexs)
    sample_index = np.random.choice(indexs, 1)
    sample_index = np.asscalar(sample_index)    
    return measurements[sample_index], images[sample_index]

def generate(n, images, measurements, bins, num_bins):
    """ 
    Return a sample of size n of angle measurements and images.
    Arguments:
    n: sample size
    images: a Python list with the information of the images.
    measurements: a Python list with the information of the angle measurements.
    bins: the bin.
    num_bins: the number of bins.    
    Returns:
    A python tuple with a sample of size n of angle measurements and images.
    """            
    image = [] 
    measu = []  # measurement 
    
    for i in range(0,n):
        b = select_bin(bins, num_bins)
        m, i = select_element_bin(b, measurements, images)
        measu.append(m)
        image.append(i)
    return measu, image     

measu, image = generate(21000, images, measurements, bins, nbins)
    

# the histogram of the data
nbins = 5

n, bins, patches = plt.hist(measu, nbins, facecolor='g', alpha=0.75)

plt.xlabel('Steering angle')
plt.ylabel('Frequency')
plt.title('Histogram of Steering angles')
plt.show()


# ## Data augmentation
# 
# The new generated data is uniform. We are going to generate more data flipping the images.

augmented_images, augmented_measurements = [], []

for image, measurement in zip(image, measu):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)

y_train = np.array(augmented_measurements)


# ## Modeling 


from keras.models import Sequential 
from keras.layers import Flatten, Dense, Activation, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


# NVIDIA deep learning model with dropout. 

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))

model.add(Conv2D(24, (5, 5) , strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5, 5) , strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5, 5) , strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3, 3) , activation="relu"))
model.add(Conv2D(64, (3, 3) , activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.60))
model.add(Dense(50))
model.add(Dropout(0.60))
model.add(Dense(10))
model.add(Dropout(0.60))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')


model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=1)


model.save('model.h5')

import gc
gc.collect()
