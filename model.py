
#------------ import Packages -----------------------
import csv
import cv2
import numpy as np
import os
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Reshape
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib
matplotlib.use('agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt




#------------ Loading the Data -----------------------


lines = []
images = []
measurements = []
correction = 0.25 #corecction value of steering for left and right camera image


with open ('data/normal1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
with open ('data/normal2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)  
    for line in reader:
        lines.append(line)
with open ('data/clockwise/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


        
for line in lines:
    random_raw = random.randint(0, 2)    
    #add left and right camera image
    for i in range(3):
        source_path = line[i]
        path_list = source_path.split(os.sep)
        filename = path_list[0].split('\\')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        
        #add the correction value for left and right camera image
        if i == 1:
            measurement = measurement + correction
         
        if i == 2: 
            measurement = measurement - correction
        
        measurements.append(measurement) 
                
        #Augmentation           
        if random_raw == i:
            
            #1. Flipping image and measurement
            aug_image = cv2.flip(image, 1) 
            aug_measurement = -measurement
            images.append(aug_image)
            measurements.append(aug_measurement) 
        
            #2. Increase the Brightness        
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
            image_hsv[:,:,2] = image_hsv[:,:,2] * ratio
            bright_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)           
            bright_measurement = measurement
            images.append(bright_image)
            measurements.append(bright_measurement) 
            
             
#save as numpy array    
X_train = np.array(images)
y_train = np.array(measurements)



#-------- The CNN-Model ---------------

model = Sequential()

# Data Preprocessing 

# 1. Normalization and Mean Centering
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# 2. Cropping image
model.add(Cropping2D(cropping=((70,25), (0,0))))
        

# modified NVIDIA Architecture  
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))    
model.add(Convolution2D(64,3,3, activation='relu'))     
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))  
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
    
    
#-------------- Training --------------
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=1)

model.save('model.h5')


#-------------- Visual Output of Training --------------------------

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
a = np.array(history_object.history['loss'])
b = np.array(history_object.history['val_loss'])
c = np.array([1, 2])
plt.figure()
plt.plot(c, a)
plt.plot(c, b)
plt.axis([1, 2, 0, 0.06])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.grid(True)
plt.savefig('figure2.jpg')
plt.show()


