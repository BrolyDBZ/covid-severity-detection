import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nibabel as nib
import cv2 as cv
import os
from IPython.display import Image
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, concatenate, Activation, BatchNormalization, \
    Input
from tensorflow.keras.models import Model
from sklearn.metrics import multilabel_confusion_matrix

"""
classification model for covid severity detection to predict the class of infection severity
 Healthy Lungs-class 0
 Infected lungs-class 1
                class 2
                class 3
the classification model is based on  resnet50 and InceptionV3 pretrained model
"""


def classification_model(input_shape):
    inputs = Input(shape=input_shape, name="input")
    googlenet = InceptionV3(include_top=False, weights="imagenet", input_tensor=inputs)
    googlenet.trainable = False
    feature = googlenet.get_layer("mixed8").output
    flattern = tf.keras.layers.Flatten()(feature)
    flattern = tf.keras.layers.Dropout(0.3)(flattern)
    Dense1 = tf.keras.layers.Dense(128, activation='relu')(flattern)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(Dense1)
    model = Model(inputs, outputs, name="classfication_inc")
    return model


def classification_resnet(input_shape):
    inputs = Input(shape=input_shape, name="input")
    res = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    res.trainable = False
    Feature = res.get_layer("conv4_block6_out").output
    flattern = tf.keras.layers.Flatten()(Feature)
    flattern = tf.keras.layers.Dropout(0.3)(flattern)
    Dense1 = tf.keras.layers.Dense(128, activation='relu')(flattern)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(Dense1)
    model = Model(inputs, outputs, name="classfication_inc")
    return model


"""
segmentation model for covid severity detection to get the mask of infection from
the covid
Unet based segmentation model uses the resnet50 pretained mmodel
"""


def conv_block(input, num_filter):
    x = Conv2D(filters=num_filter, kernel_size=3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=num_filter, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def decoder(input, skip_feature, num_filter):
    x = Conv2DTranspose(filters=num_filter, kernel_size=2, strides=2, padding="same")(input)
    x = concatenate([x, skip_feature])
    x = conv_block(x, num_filter)
    return x


def segmentation_model(input_shape):
    inputs = Input(shape=input_shape, name="input")
    res = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    res.trainable = False
    s1 = res.get_layer("input").output
    s2 = res.get_layer("conv1_relu").output
    s3 = res.get_layer("conv2_block3_out").output
    s4 = res.get_layer("conv3_block4_out").output

    feature = res.get_layer("conv4_block6_out").output

    d1 = decoder(feature, s4, 512)
    d2 = decoder(d1, s3, 256)
    d3 = decoder(d2, s2, 128)
    d4 = decoder(d3, s1, 64)

    outputs = Conv2D(4, (1, 1), padding="same", activation="softmax")(d4)
    model = Model(inputs, outputs, name="Unet_res")
    return model
