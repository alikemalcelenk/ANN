#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 23:33:00 2020

@author: alikemalcelenk
"""

# L - LAYER NEURAL NETWORK

#DATASETİ HAZIRLIYORUZ 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


x_l = np.load('./X.npy')
Y_l = np.load('./Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')


# Join a sequence of arrays along an row axis.
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)  #(410, 64, 64)
print("Y shape: " , Y.shape)  #(410, 1)

# Then lets create x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape) #X train flatten (348, 4096)
print("X test flatten",X_test_flatten.shape) #X test flatten (62, 4096)

#artificalNeuralNetwork.py ve logisticRegression.py de .T larını almıştık. bunda almıyoruz
x_train = X_train_flatten
x_test = X_test_flatten
y_train = Y_train
y_test = Y_test

print("x train: ",x_train.shape)  #(4096, 348)
print("x test: ",x_test.shape) #(4096, 62)
print("y train: ",y_train.shape) #(1, 348)
print("y test: ",y_test.shape) #(1, 62)


# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library


def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    # classifier.add(Dense(  =  hidden layer ekler.
    # units = nodes
    # kernel_initializer = weight leri initilaze etmemize yarıyor.
    # activation = activation func.
    # input_dim = 4096
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # sigmoid cuz output
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # adam=adaptive momentum
    # loss = 'binary_crossentropy'   loss funcımız. Logistic regression da kullandığımızın aynısı.
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
# epochs = number of iteration
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
# cv = cross validation
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))
