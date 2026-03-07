#-------------------------------------------------------------------------
# AUTHOR: Nicholas Tran
# FILENAME: knn.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

#Loop your data to allow each instance to be your test set
error_count = 0
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    for sample in db:
        if sample is not i:
            features = []
            for value in sample[:-1]:
                features.append(float(value))
            X.append(features)

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    for sample in db:
        if sample is not i:
            if str(sample[-1]).lower() == 'spam':
                Y.append(1.0)
            else:
                Y.append(0.0)

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = []
    for value in i[:-1]:
        testSample.append(float(value))

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    # clf =
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if str(i[-1]).lower() == 'spam':
        true_label = 1.0
    else:
        true_label = 0.0
    if class_predicted != true_label:
        error_count += 1

#Print the error rate
#--> add your Python code here
print(f'LOO-CV error rate = {error_count / len(db):.2f}')




