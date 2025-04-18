#Importing all the modules to be used in the project


import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation,BatchNormalization,add
from tensorflow.keras.models  import Model,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
import os

#Here we are loading the VGG16 model and defining our model and its layers which are going to be used to detect pneumonia.
vgg=VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3)layers)

for layer in vgg.layer:
    layer.trainable=False              #making all layers non-trainable

x=flatten()(vgg.output)

predictions=
Dense(2,activation='softmax')(x)

model=model(inputs=vgg.input,output=predictions)
model.summary()

#Here we are initializing the data generator for training 
# the model and we are loading the VGG16 model with ImageNet weights without the Fully Connected Layers.

target_shape=(224,224)

train_dir="train"
val_dir="val"          #--directories for data
test_dir="test"

vgg=VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))

for layer in vgg.layers:
    layer.trainable=False              #making all the layers non-trainable


#Here defining the model and its layers

x=Flatten()(vgg.output)
predictions=
Dense(2,activation='softmax')(x)
model=model(inputs=vgg.input,outputs=predictions)
model.summary()

#here we are getting our traininand testing data

train_gen=ImageDataGenerator(rescale=1/255.0,

horizontal_flip=true,zoom_range=0.2,shear_range=0.2)

#making the data loader for validation data 
test_gen=ImageDataGenerator(rescale=1/255.0)

#function to make iterable object for training
train_data_gen=train_gen.flow_from_directory(train_dir,target_shape.batch_size=16,class_mode='categorical')

#function to make iterable object for training
test_data_gen=train_gen.flow_from_directory(test_dir,target_shape.batch_size=16,class_mode='categorical')

#Here We are fitting our training data into our model. Also, we are passing the test data and generating the output.

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist=model.fit_generation(train_data_gen,
                         steps_per_epoch=20,
                         epochs=20,
validation_data=test_data_gen,validation_steps=10)

#Here we are plotting an accuracy and loss curve.

plt.style.use("ggplot")
plt.figure()
plt.plot(hist.history["loss"],
label="train_loss")
plt.plot(hist.history["val_loss"],
label="val_loss")
plt.plot(hist.history["accuracy"],
label="train_acc")
plt.title("Model Training")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("epochs.png")
