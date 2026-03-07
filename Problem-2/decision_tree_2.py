#-------------------------------------------------------------------------
# AUTHOR: Nicholas Tran
# FILENAME: decision_tree_2.py
# SPECIFICATION: Run train 3 decision tree models on 3 different data sets and calculate thier average accuracy over 10 iterations for each set.
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    dbTraining = pd.read_csv(ds).values.tolist()

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    
    Feature_Mapping = {
        'Young': 1,
        'Prepresbyopic': 2,
        'Presbyopic': 3,
        'Myope': 1,
        'Hypermetrope': 2,
        'Yes': 1,
        'No': 0,
        "Normal": 1,
        "Reduced": 2
    }

    for data in dbTraining:
        features = [Feature_Mapping[data[0]], Feature_Mapping[data[1]], Feature_Mapping[data[2]], Feature_Mapping[data[3]]]
        X.append(features)

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for data in dbTraining:
        Y.append(Feature_Mapping[data[4]])

    #Loop your training and test tasks 10 times here
    model_accuracies = []
    for i in range (10):
        correct_prediction = 0
       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> addd your Python code here
       # clf =
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here
        dbTest = []
        df_test = pd.read_csv('contact_lens_test.csv')
        for _, row in df_test.iterrows():
            dbTest.append(row.tolist())

        for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            features = [Feature_Mapping[data[0]], Feature_Mapping[data[1]], Feature_Mapping[data[2]], Feature_Mapping[data[3]]]
            class_predicted = clf.predict([features])[0]
            
           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            true_label = Feature_Mapping[data[4]]
            if class_predicted == true_label:
                correct_prediction += 1
        accuracy = correct_prediction / len(dbTest)
        model_accuracies.append(accuracy)

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    average_accuracy = sum(model_accuracies) / len(model_accuracies)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"The final accuracy of the model when training on {ds} is: {average_accuracy:.2f}")




