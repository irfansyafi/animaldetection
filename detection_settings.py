import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
import PIL.Image
import pathlib


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


#To get the category names and put it in an array
category = [entry.name for entry in path_object.iterdir() if entry.is_dir()]
print(category)

#Exploring the dataset and do the necessary preprocessing

def explore_img_data(path_object):
    #explore example for bear
    bear = list(path_object.glob('Bear/*'))
    PIL.Image.open(str(bear[0])).show() # this may open a file extension of .png instead of .jpg
    
explore_img_data(path_object)

def set_train_dataset():
    # We already have a dataset in our local directory.
    # Use the tf.keras.utils.image_dataset_from_directory and path_object to get the dataset

    # define the batch size for the dataset. Batch size is the number of images in one batch. For example, batch_size = 30. There is only 30 images in a batch.
    global batch_size, img_height, img_width
    batch_size = 30

    # define the image height and width
    # most of image dimension is width x height and it is in pixel
    # check your dataset 
    # by defining the width and height of the image, we actually are rescaling the images in pixel

    img_width = 224
    img_height = 224

    # Below will get the image dataset and turn it into a batch dataset stored in the train_dataset variable
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        path_object,
        #validation_split=0.2, # If validation split = 0.2 means it is splitting the dataset into 80%:20% training and validation set. In this case we don't use the validation parameter because we already have the test dataset in our local directory
        #subset="training",  # if subset is set then the validation must be set as well.
        #color_mode='rgb', # it is default 'rgb'
        seed = 123,
        image_size=(img_height,img_width),
        batch_size= batch_size
    )
    print(f"The amount of batch : {len(train_dataset)}")
    
    
    return train_dataset

# Iterate through the batches and visualize the images one by one
'''
for images, labels in train_dataset:
    for image in range(2):
        # Display or process the individual image as needed
        PIL.Image.fromarray((image.numpy() * 255).astype('uint8')).show() # image.numpy() * 255 because assuming the train dataset in range 0 to 1 value
'''

''' Testing the loop and how in matplotlib 
for images, labels in train_dataset:
    # Access the first image and its label from the batch
    image = images[0]
    label = labels[0]

    # Display or process the individual image as needed
    plt.imshow(image.numpy().astype("uint8"))  # Assuming images are in uint8 format
    plt.title(label.numpy())  # Display the numeric label
    plt.show()

    # Exit the loop after displaying the first image and label
    break
'''

''' Testing to open the picture using PIL for learning purposes '''
'''
for images,labels in train_dataset:
    image = images[0]
    label = labels[0]
    labelname = category[label]
    
    pil_image = PIL.Image.fromarray((image.numpy()).astype('uint8'))
    
    pil_image.show()
    
    print(f"Label : {label}, {labelname}")
    
    break
'''



''' Learning purposes 
# Iterate through the batches and visualize the images in the first batch only
for batch_num, (images, labels) in enumerate(train_dataset):
    if batch_num < 100:
        # Loop through each image in the first batch
        for i, label in enumerate(labels.numpy()):
            class_name = category[label]
            print(f"Sample {i+1}: Numeric Label: {label}, Class Name: {class_name}")
    else:
        break  # Exit the loop after processing the first batch

'''

# Assuming you have already created the train_dataset
# You can use the as_supervised=True argument to load the data as (image, label) pairs

# Iterate through the batches and visualize the images in each batch

''' 
    Example 1 below
for images, labels in train_dataset:
    # Loop through each image in the batch
    for image in images:
        # Display or process the individual image as needed
        plt.imshow(image.np().astype("uint8"))  # Assuming images are in uint8 format
        plt.show()

    Example 2 below
# Iterate through the batches and visualize the images in the first batch only
for batch_num, (images, labels) in enumerate(train_dataset):
    if batch_num == 0:
        # Loop through each image in the first batch
        for image,label in zip(images,labels):
            # Display or process the individual image as needed
            plt.imshow(image.numpy().astype("uint8"))  # Assuming images are in uint8 format
            plt.title(label.numpy())
            plt.show()
    else:
        break  # Exit the loop after processing the first batch
'''

