#-------------------------------------------------------------------------
# AUTHOR: Nicholas Tran
# FILENAME: naive_bayes.py
# SPECIFICATION: Read the training data set and train a naive bayes classifier. Then read the test data set and classify the test samples. Use a classification confidence threshold of >= 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 day and 5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature_map = {'Cool': 1, 'Mild': 2, 'Hot': 3}
humidity_map = {'Normal': 1, 'High': 2}
wind_map = {'Weak': 1, 'Strong': 2}

X = []
for sample in dbTraining:
    X.append([
        outlook_map[sample[1]],
        temperature_map[sample[2]],
        humidity_map[sample[3]],
        wind_map[sample[4]]
    ])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
class_map = {'Yes': 1, 'No': 2}
inverse_class_map = {1: 'Yes', 2: 'No'} # For printing the predicted label later

Y = []
for sample in dbTraining:
    Y.append(class_map[sample[5]])

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print(f'{"Day":<6}{"Outlook":<10}{"Temperature":<13}{"Humidity":<10}{"Wind":<8}{"PlayTennis":<12}{"Confidence":<10}')

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for sample in dbTest:
    test_vector = [[
        outlook_map[sample[1]],
        temperature_map[sample[2]],
        humidity_map[sample[3]],
        wind_map[sample[4]]
    ]]
    probas = clf.predict_proba(test_vector)[0]
    predicted_class_index = probas.argmax()
    confidence = probas[predicted_class_index]
    predicted_label_num = clf.classes_[predicted_class_index]
    predicted_label = inverse_class_map[predicted_label_num]

    if confidence >= 0.75:
        print(
            f'{sample[0]:<6}{sample[1]:<10}{sample[2]:<13}{sample[3]:<10}{sample[4]:<8}{predicted_label:<12}{confidence:<10.2f}'
        )

