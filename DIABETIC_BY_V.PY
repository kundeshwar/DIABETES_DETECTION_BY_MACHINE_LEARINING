#this machine learning programe is very useful to predict person haveing diabetic or not 
# this is also classifications problem you will use logistic model 
#  first you have to install all useful labrary that are numpy, pandas, matplotlib.pyplot, scikit learn
#follow below processier
#Diabetes data
#data pre-processing
#train-test-split
#accurancy score
#support vector machine classifier
#prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#-----------------------------------------------this is data anaylsis 
#PIMA Diabetes Dataset
data = pd.read_csv("diabetes.csv")
print(data.columns)
print(data.head(10))
print(data["Outcome"].unique())
print(data.shape)
print(data.describe())
print(data["Outcome"].value_counts())
print(data.groupby("Outcome").mean())
#data = pd.read_csv("diabetes.csv", header=None)
#print(data)
#-----------------------------------------------this is data separtions 
y = data["Outcome"]
print(y)

x = data.drop(columns="Outcome", axis=1)
print(x)
#--------------------------------------------------standardized data for acurate result
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
print(standardized_data)
x_new = standardized_data

#---------------------------------------------------split data train test data
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, stratify=y, random_state=2)
print(x_train.shape, x_test.shape, x_new.shape)
print(y.shape, y_train.shape, y_test.shape)

#--------------------------------------------------train model and use of model
classifer = svm.SVC(kernel='linear')
classifer.fit(x_train, y_train)
#----------------------------------------------------Evaluations
x_train_prediction = classifer.predict(x_train)
training_data_accurancy = accuracy_score(x_train_prediction, y_train)
print(training_data_accurancy)

x_test_prediction = classifer.predict(x_test)
test_data_accurancy = accuracy_score(x_test_prediction, y_test)
print(test_data_accurancy)
#--------------------------------------------------predictions
y_new_predictions = classifer.predict(x_test)
print(y_new_predictions,"this is my predictgions data", y_test, "this is my oroginal data ")

#---------------------------------------this is my accurancy of model 
accurancy_of_our_model = accuracy_score(y_new_predictions, y_test)
print(accurancy_of_our_model)

