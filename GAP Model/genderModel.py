import csv
import numpy as np
import Tkinter as tk
import tkFileDialog as fd

## Step 1: loading .csv training dataset
##root = tk.Tk()
##root.withdraw()
##file_path = fd.askopenfilename()
file_path = '../data/csv/train.csv'
rawData = csv.reader(open(file_path,'rb'))
colnames = rawData.next()
## 0: 'PassengerId', 1: 'Survived', 2: 'Pclass', 3: 'Name',
## 4: 'Sex', 5: 'Age', 6: 'SibSp', 7: 'Parch', 8: 'Ticket',
## 9: 'Fare', 10: 'Cabin', 11: 'Embarked'
data = []
for row in rawData:
    data.append(row)
data = np.array(data)

## Step 2: split the training dataset into two parts: training set and cv set
trainingSetSize = int(len(data)*3/4)
trainingSet = data[0:trainingSetSize]
crossVSet = data[trainingSetSize:len(data)]

## Step 3: applying the gender model to check if gender is a good feature
women_survived = np.sum(trainingSet[trainingSet[0::,4]=='female',1].astype(np.float))
women_onboard = np.size(trainingSet[trainingSet[0::,4]=='female',1].astype(np.float))
men_survived = np.sum(trainingSet[trainingSet[0::,4] == 'male',1].astype(np.float))
men_onboard = np.size(trainingSet[trainingSet[0::,4] == 'male',1].astype(np.float))

women_ratio = women_survived / women_onboard
men_ratio = men_survived / men_onboard

print "The ratio of women survived: " + str(women_ratio)
print "The ratio of men survived: " + str(men_ratio)

## Step 4: check the validity of the gender model on cv set
cv_survived = []
for cv_row in crossVSet:
    if cv_row[4] == 'female':
        cv_survived.append([cv_row[0],cv_row[1],'1'])
    else:
        cv_survived.append([cv_row[0],cv_row[1],'0'])
cv_survived = np.array(cv_survived)

correct = 0
for cv_row in cv_survived:
    if cv_row[1] == cv_row[2]:
        correct += 1
total = len(cv_survived)
accuracy = float(correct) / float(total)

print "The accuracy of the gender model on cv set is: " + str(accuracy)

