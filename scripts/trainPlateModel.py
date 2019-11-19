import math
import numpy as np
import re
import cv2
from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
import os
import pickle

from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend

NUMBER_OF_LABELS = 36*36
CONFIDENCE_THRESHOLD = 0.01
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

#Load Dataset
def loadDataset(PATH):
    labels = os.listdir(PATH)

    dataset = np.zeros((3*len(labels), 2), dtype=object)

    #Read labels and create y value with unicode
    for i in range(len(labels)):
        image = cv2.imread(PATH+labels[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dataset[3*i,0] = cv2.resize(image[750:1050, :], (64,64))
        if ord(labels[i][0]) > 58:
            dataset[3*i,1] = (ord('P')-55)*36 + ord(labels[i][0]) - 55
        else:
            dataset[3*i,1] = (ord('P')-55)*36 + ord(labels[i][0]) - 48


        dataset[3*i+1,0] = cv2.resize(image[1250:1550, 0:300], (64, 64))
        if ord(labels[i][1]) > 58:
            a = ord(labels[i][1]) - 55
        else:
            a = ord(labels[i][1]) - 48

        if ord(labels[i][2]) > 58:
            b = ord(labels[i][2]) - 55
        else:
            b = ord(labels[i][2]) - 48
        dataset[3*i+1,1] = a*36 + b


        dataset[3*i+2,0] = cv2.resize(image[1250:1550, 300:600], (64, 64))
        if ord(labels[i][3]) > 58:
            a = ord(labels[i][3]) - 55
        else:
            a = ord(labels[i][3]) - 48

        if ord(labels[i][4]) > 58:
            b = ord(labels[i][4]) - 55
        else:
            b = ord(labels[i][4]) - 48
        dataset[3*i+2,1] = a * 36 + b

    print("Loaded {} images from folder".format(len(labels)))

    # Shuffle the dataset
    np.random.shuffle(dataset)

    # Split data into x and y data
    X_dataset_orig = np.array([data[0] for data in dataset[:]])
    Y_dataset_orig = np.array([[data[1]] for data in dataset]).T
        
    # Normalize X (images) dataset
    X_dataset = X_dataset_orig/255.
    X_dataset = np.expand_dims(X_dataset, axis=3)

    # Convert Y dataset to one-hot encoding
    Y_dataset = convert_to_one_hot(Y_dataset_orig, NUMBER_OF_LABELS).T

    #Print info about data
    print("Total examples: {:d}\nTraining examples: {:f}\nTest examples: {:f}".
        format(X_dataset.shape[0],
                math.ceil(X_dataset.shape[0] * (1-VALIDATION_SPLIT)),
                math.floor(X_dataset.shape[0] * VALIDATION_SPLIT)))
    print("X shape: " + str(X_dataset.shape))
    print("Y shape: " + str(Y_dataset.shape))



    #X_dataset = np.stack(X_dataset)
    return X_dataset, Y_dataset

def trainModel(X_dataset, Y_dataset):
    conv_model = models.Sequential()
    conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(64, 64, 1)))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Flatten())
    conv_model.add(layers.Dropout(0.5))
    conv_model.add(layers.Dense(512, activation='relu'))
    conv_model.add(layers.Dense(36*36, activation='softmax'))


    conv_model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=1e-4),
                    metrics=['acc'])

    conv_model.summary()

    #Train CNN
    reset_weights(conv_model)
    history_conv = conv_model.fit(X_dataset, Y_dataset, 
                                validation_split=VALIDATION_SPLIT, 
                                epochs=50, 
                                batch_size=16)

    return conv_model


#Resets CNN training parameters
def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
            



#Save model to pickel 



if __name__ == "__main__":
    x,y = loadDataset("/home/fizzer/enph353_ws/src/enph353/enph353_gazebo/media/materials/textures/dataset/")
    model = trainModel(x,y)
    pickle.dump({model}, open('model.p','wb'))
