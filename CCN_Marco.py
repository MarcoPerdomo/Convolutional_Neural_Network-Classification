
# Convolutional Neural Network by MARCO PERDOMO

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# The data preprocessing is made by separating the dog and cat pictures in different folders

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) 
#The number of filters is 32, and that means that we will obtain 32 feature maps.
# 3 is our feature detector size: 3x3 
# We use Relu or Rectifier linear function to add non-linearity, which is required for a classification problem

# step 2 -  Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# This step allows use to reduce the size of the nodes for the flattening step
# It also helps maintain the spacial invariance 
# Helps use reduce the complexity and time of the algorithm

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu')) #The input shape is the size of the pooling maps
#We don't actually need to specify the shape, we only do that when there is nothing before
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flatteing 
classifier.add(Flatten())
# When we apply convolution, pooling and flattening the program understands the spacial structure of the image
# If we apply flattening to the picture without the previous steps, it's only pixels information, nothing else

# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
# We choose a number of nodes in the hidden layer of 128. It doesn't need to be too small but not too big
# It is a good practice to use a power of two number

classifier.add(Dense(units = 1, activation = 'sigmoid')) 
# Sigmoid activation function for a binary outcome, cause we predicting the probability of one class
# This is because we have the dogs and cats separated in different folders

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

# Image preprocessign or image augmentation prevents overfitting due to few training samples
# Applies different random processing like shifting, turning, etc... to augment the training images

from keras.preprocessing.image import ImageDataGenerator

#From Keras documentation:
train_datagen = ImageDataGenerator(
        rescale=1./255, # Feature Scaling so that pixels with values from 1-255 have values from 0 to 1
        shear_range=0.2, # Transvection transformation (Geometrical transformation)
        zoom_range=0.2, # Random zooming
        horizontal_flip=True) 

test_datagen = ImageDataGenerator(rescale=1./255) # Feature scaling of test set

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32, #Number of images after which the weights will be updated
                                                 class_mode='binary') #Binary or more categories  

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
 

classifier.fit_generator(training_set,
                         steps_per_epoch=8000, #Number of images in the training set
                         epochs=20, #The more we have, the longer it will take
                         validation_data=test_set,
                         validation_steps=2000) #Number of images in our test set


# Part 3 -  Making a single prediction
import numpy as np # We need a function from numpy to process the image
from keras.preprocessing import image 
test_image = image.load_img('dataset/single_prediction/Me.jpg', target_size=(64, 64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
# The predict function cannot accept a single accept by itself, it only accepts inputs in a batch
# That's why we add a new dimension with the batch, even if it is only for a single prediction

result = classifier.predict(test_image) #This gives either a 1 or 0 
training_set.class_indices # This gives us the mapping of the predictions. i.e. Dogs is 1 and cats is 0
if result[0][0] == 1:
    prediction = 'dog'
else:
        prediction = 'cat'


