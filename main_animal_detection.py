import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt
import keras
from keras import layers,datasets,models
#import detection_settings

# First, we create a function to load the data into our project from the local directory.
def load_img_data():
    path_train_dataset = "C:/Users/HTC/Desktop/PORTFOLIO AND GITHUB/Animal Detection (Computer Vision)/train"  
    path_config = tf.keras.utils.get_file(origin='',fname=path_train_dataset)  #origin parameter is set to an empty string (''), which means you are not actually downloading a file from a remote URL but rather trying to obtain a local file with the path specified in fname. The provided path appears to be a local path on your system.
    path_object = pathlib.Path(path_config) # creating the Path object using pathlib module to make it easier
    image_count = len(list(path_object.glob('*/*.jpg'))) # the */ find any subdirectory /*.jpg find matches file that have jpg extension
    print(f"Total image count : {image_count}")
    
    return path_object
    # return the path object to be define in a global variable to be accessible to other function.

path_object = load_img_data()

#def run_code():
#To get the category names and put it in an array
category = [entry.name for entry in path_object.iterdir() if entry.is_dir()]

print(category)

batch_size = 30
img_height = 224
img_width = 224

def create_dataset():
    global train_dataset,validation_dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        path_object,
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode='rgb',
        image_size=(img_height,img_width),
        batch_size=batch_size
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        path_object,
        validation_split=0.2,
        subset="validation",
        seed = 123,
        color_mode='rgb',
        image_size=(img_height,img_width),
        batch_size=batch_size
    )
    
    print(f"The amount of batch in train_dataset : {len(train_dataset)}")
    print(f"The amount of batch in validation_dataset : {len(validation_dataset)}")
    
    return train_dataset,validation_dataset

train_DS, validation_DS = create_dataset()

classname = train_DS.class_names

print(classname)

# to open the image available in the train_DS or validation_DS, we need to use for loop to iterate 

for images, labels in validation_DS:
    image = images[0]
    label = labels[0]
    labelname = category[label]
    
    pil_image = PIL.Image.fromarray((image.numpy()).astype('uint8'))
    
    pil_image.show()
    
    print(f"Label : {label}, {labelname}")
    
    break # if not break, it will iterate all the image to open


# Data transformation for standardization when training a neural network. In this case, the CNN model.

# {CODE HERE STILL IN PROGRESS}


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_DS.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = validation_DS.cache().prefetch(buffer_size=AUTOTUNE)

# Create the base for CNN
''' --- Experiment 2 --- '''

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(80) #input the number of class
])



'''
The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.

As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size. 
If you are new to these dimensions, color_channels refers to (R,G,B). 
In this example, you will configure your CNN to process inputs of shape (32, 32, 3), which is the format of CIFAR images. 
You can do this by passing the argument input_shape to your first layer.
'''

''' --- Experiment 1 failed due to rescaling added has cause problem when compiling ----

model = models.Sequential()
model.add(layers.Rescaling(1./255)) # add rescaling layer for standardization
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.build(input_shape=(224,224,3))
model.summary()
'''

'''
Add Dense layers on top
To complete the model, you will feed the last output tensor from the convolutional base (of shape (4, 4, 64)) into one or more Dense layers to perform classification. 
Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor. 
First, you will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top. 
CIFAR has 10 output classes, so you use a final Dense layer with 10 outputs.
'''

''' Continous of experiment 1 --- 


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(80))

model.summary()

'''
'''
Compile and train the data
'''
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


history = model.fit(train_DS,validation_data=validation_DS,epochs=5)

