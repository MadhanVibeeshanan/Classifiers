# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 17:57:40 2018

@author: madha
"""

from scipy.misc import imread
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import *
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import time
#***************************************************************************************************************************************

facetrain = glob.glob("C:/Users/madha/Documents/Sec Sem/Machine Learning/Assignment 3/MIT-CBCL-Face-dataset/train/face/*.pgm")
#facetrainmatrix =(pd.DataFrame(np.random.randint(low=0, high=1, size=(361, len(facetrain)))))

non_facetrain = glob.glob("C:/Users/madha/Documents/Sec Sem/Machine Learning/Assignment 3/MIT-CBCL-Face-dataset/train/non-face/*.pgm")      
#non_facetrainmatrix =(pd.DataFrame(np.random.randint(low=0, high=1, size=(361, len(non_facetrain)))))

facetest = glob.glob("C:/Users/madha/Documents/Sec Sem/Machine Learning/Assignment 3/MIT-CBCL-Face-dataset/test/face/*.pgm")      
#facetestmatrix =(pd.DataFrame(np.random.randint(low=0, high=1, size=(361, len(facetest)))))

non_facetest = glob.glob("C:/Users/madha/Documents/Sec Sem/Machine Learning/Assignment 3/MIT-CBCL-Face-dataset/test/non-face/*.pgm")      
#non_facetestmatrix =(pd.DataFrame(np.random.randint(low=0, high=1, size=(361, len(non_facetest)))))

def pixel_conversion(csv_file):
    start_time = time.time()
    matrix1 = []
    for filename in csv_file:
        im = imread(filename)
        matrix1.append(im.ravel())
    print("Appeding Done ")  
    matrix1 = np.matrix(matrix1)
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    return matrix1

facetrainmatrix = pixel_conversion(facetrain)
print("Finished loading Face Train Marix")
facetrainmatrix = np.c_[facetrainmatrix, np.ones((facetrainmatrix.shape[0]))]
facetrainmatrix = pd.DataFrame(facetrainmatrix)

non_facetrainmatrix = pixel_conversion(non_facetrain)
print("Finished loading Non-Face Train Matrix")
non_facetrainmatrix = np.c_[non_facetrainmatrix, np.zeros((non_facetrainmatrix.shape[0]))]
non_facetrainmatrix = pd.DataFrame(non_facetrainmatrix)

facetestmatrix = pixel_conversion(facetest)
#facetestmatrix = np.matrix(facetestmatrix)
facetestmatrix = np.c_[facetestmatrix, np.ones((facetestmatrix.shape[0]))]
facetestmatrix = pd.DataFrame(facetestmatrix)
print("Finished loading Face Test Matrix")

non_facetestmatrix = pixel_conversion(non_facetest)
#non_facetestmatrix = np.matrix(non_facetestmatrix)
non_facetestmatrix = np.c_[non_facetestmatrix, np.zeros((non_facetestmatrix.shape[0]))]
non_facetestmatrix = pd.DataFrame(non_facetestmatrix)
print("Finished loading Non-Face Test Matrix")


#Training Dataset
train_dataset = np.concatenate((facetrainmatrix, non_facetrainmatrix))
train_dataset = pd.DataFrame(train_dataset)

#Splitting  Traing Dataset
x_train = train_dataset.iloc[: , :-1].values
y_train = train_dataset.iloc[: , -1].values

#Adding Bias Term for Traning Dataset
x_train = np.c_[np.ones((x_train.shape[0])),x_train]

#Testing Dataset
test_dataset = np.concatenate((facetestmatrix, non_facetestmatrix))
test_dataset = pd.DataFrame(test_dataset)

#Splitting Testing Dataset
x_test = test_dataset.iloc[: , :-1].values
y_test = test_dataset.iloc[: ,-1].values

# Adding Bias Term for Testing Dataset
x_test = np.c_[np.ones((x_test.shape[0])),x_test]

# W value
w = np.random.uniform(size=(x_test.shape[1],))
w = np.random.uniform(low =0, high = 1,size=(x_test.shape[1],))

nEpoch = 2
alpha = 0.1
lamda = 0.5

#Sigmoid Function
def sigmoidfunction(z):
    return 1.0 / (1.0 + np.exp(-z))

for epoch in np.arange(0, nEpoch):
    print("Initiated")
    for i in range(len(x_train)):
        hypo = sigmoidfunction(x_train.dot(w))
        error = (hypo - y_train)
        reg = (lamda * (w))
        gradient = ((x_train.T.dot(error))-(reg))
        if (lamda!=0):
            gradient[0] = np.sum(error)
        w = w-alpha*gradient
    print("nEpoch repeatation ", epoch)

ypred = sigmoidfunction(x_test.dot(w))

#confusion matrix
conf_matrix =confusion_matrix(y_test, ypred)

#Goals
accuracy = accuracy_score(y_test, ypred)
precision = precision_score(y_test, ypred)
recall = recall_score(y_test, ypred)
f1 = f1_score(y_test, ypred)

print("Accuray :", accuracy, "\nPrecision", precision, "\nRecall :", recall, "\nF1 ",f1)

fpr, tpr, threshold = metrics.roc_curve(y_test, ypred)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'green')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
