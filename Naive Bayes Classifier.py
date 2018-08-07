# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:22:00 2018

@author: madha
"""
from scipy.misc import imread
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import *
from sklearn import metrics
import math
import collections
import matplotlib.pyplot as plt
import time

facetrain = glob.glob("C:/Users/madha/Documents/Sec Sem/Machine Learning/Assignment 3/MIT-CBCL-Face-dataset/train/face/*.pgm")
non_facetrain = glob.glob("C:/Users/madha/Documents/Sec Sem/Machine Learning/Assignment 3/MIT-CBCL-Face-dataset/train/non-face/*.pgm")      
facetest = glob.glob("C:/Users/madha/Documents/Sec Sem/Machine Learning/Assignment 3/MIT-CBCL-Face-dataset/test/face/*.pgm")      
non_facetest = glob.glob("C:/Users/madha/Documents/Sec Sem/Machine Learning/Assignment 3/MIT-CBCL-Face-dataset/test/non-face/*.pgm")      


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
facetrainmatrix = pd.DataFrame(facetrainmatrix)

non_facetrainmatrix = pixel_conversion(non_facetrain)
print("Finished loading Non-Face Train Matrix")
non_facetrainmatrix = pd.DataFrame(non_facetrainmatrix)

facetestmatrix = pixel_conversion(facetest)
facetestmatrix = np.c_[facetestmatrix, np.ones((facetestmatrix.shape[0]))]
facetestmatrix = pd.DataFrame(facetestmatrix)
print("Finished loading Face Test Matrix")

non_facetestmatrix = pixel_conversion(non_facetest)
non_facetestmatrix = np.c_[non_facetestmatrix, np.zeros((non_facetestmatrix.shape[0]))]
non_facetestmatrix = pd.DataFrame(non_facetestmatrix)
print("Finished loading Non-Face Test Matrix")

train_dataset = np.concatenate((facetrainmatrix, non_facetrainmatrix))
train_dataset = pd.DataFrame(train_dataset)

test_dataset = np.concatenate((facetestmatrix, non_facetestmatrix))
test_dataset = pd.DataFrame(test_dataset)

x_test = test_dataset.iloc[: , :-1].values
y_test = test_dataset.iloc[: ,-1].values
    

def probability_matrices(matrix, prob_matrix):
    for i in matrix:
        count = collections.Counter()
        for j in (matrix[i]):
            count[j] += 1
        for k in np.arange(0, 256):
            prob_matrix[i][k] = count[k]/len(matrix)
        print("Finished ",i)
    return prob_matrix

face_probability_matrix = ((pd.DataFrame(np.random.uniform(low =0, high = 1, size=(256, 361)))))
non_face_probability_matrix = ((pd.DataFrame(np.random.uniform(low =0, high = 1, size=(256, 361)))))            

face_probability_matrix = probability_matrices(facetrainmatrix, face_probability_matrix)
print("Finished Face Images probability Matrix")

non_face_probability_matrix = probability_matrices(non_facetrainmatrix, non_face_probability_matrix)
print("Finished Non-Face Images probability Matrix")

y_face_testvalue =[]
y_non_face_testvalue = []

def finding(j, k, probability_mul, matrix):
    probability_finding = matrix[j][k]
    if(probability_finding == 0.0):
        probability_finding = smoothing(probability_finding, matrix)
    probability_mul = float(probability_mul + math.log(probability_finding)) 
    return probability_mul

def smoothing(value, matrix):
    return ((value + 1) / len(matrix))   
    
for i in range(len(test_dataset)):  
    print(i)
    face_probability_mul = 0
    non_face_probability_mul = 0
    for j in range((x_test.shape[1])):
        k = x_test[i,j]
        face_probability_mul = finding(j, k, face_probability_mul, face_probability_matrix)   
        non_face_probability_mul = finding(j , k ,non_face_probability_mul, non_face_probability_matrix) 
    y_face_testvalue.append((face_probability_mul) + (math.log (len(facetrainmatrix)/(len(train_dataset)))))
    y_non_face_testvalue.append((non_face_probability_mul) + (math.log (len(non_facetrainmatrix)/(len(train_dataset)))))

def pred():
    for i in np.arange(0, len(test_dataset)):
        if(y_face_testvalue[i] > y_non_face_testvalue[i]):
            ypred.append(1)
        else:
            ypred.append(0)
            
ypred = []
pred()

conf_matrix =confusion_matrix(y_test, ypred)

accuracy = accuracy_score(y_test, ypred)
precision = precision_score(y_test, ypred)
recall = recall_score(y_test, ypred)
f1 = f1_score(y_test, ypred)

print("Accuray :", accuracy, "\nPrecision", precision, "\nRecall :", recall, "\nF1 ",f1)

fpr, tpr, threshold = metrics.roc_curve(y_test, ypred)
roc_auc = metrics.auc(fpr, tpr)
print(tpr, fpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'black', label = 'Accuracy = %0.2f' %roc_auc)
plt.legend(loc = 'upper left')
plt.plot([0, 1], [0, 1],'r-')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



