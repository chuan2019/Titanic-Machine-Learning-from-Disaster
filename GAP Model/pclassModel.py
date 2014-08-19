import csv
import numpy as np
import Tkinter as tk
import tkFileDialog as fd

## Step 1: loading .csv training dataset
root = tk.Tk()
root.withdraw()
file_path = fd.askopenfilename()
rawData = csv.reader(open(file_path,'rb'))
colnames = rawData.next()
## 0: 'PassengerId', 1: 'Survived', 2: 'Pclass', 3: 'Name',
## 4: 'Sex', 5: 'Age', 6: 'SibSp', 7: 'Parch', 8: 'Ticket',
## 9: 'Fare', 10: 'Cabin', 11: 'Embarked'
data = []
for row in rawData:
    data.append(row)
data = np.array(data)

## Omitting NAs in Pclass field
havePclass = data[0::,2] != ''
pcData = data[havePclass,0::]
data = pcData

## Step 2: split the training dataset into two parts: training set and cv set
trainingSetSize = int(len(data)*3/4)
trainingSet = data[0:trainingSetSize]
crossVSet = data[trainingSetSize:len(data)]

## Step 3: applying the Pclass model to check if Pclass is a good feature
survived1 = np.sum(trainingSet[trainingSet[0::,2]=='1',\
                                    1].astype(np.float))
onboard1 = np.size(trainingSet[trainingSet[0::,2]=='1',\
                                    1].astype(np.float))
survived2 = np.sum(trainingSet[trainingSet[0::,2]=='2',\
                                    1].astype(np.float))
onboard2 = np.size(trainingSet[trainingSet[0::,2]=='2',\
                                    1].astype(np.float))
survived3 = np.sum(trainingSet[trainingSet[0::,2]=='3',\
                                    1].astype(np.float))
onboard3 = np.size(trainingSet[trainingSet[0::,2]=='3',\
                                    1].astype(np.float))

ratio1 = survived1 / onboard1
ratio2 = survived2 / onboard2
ratio3 = survived3 / onboard3

print "The ratio of 1st class passenger survived: " + str(ratio1)
print "The ratio of 2nd class passenger survived: " + str(ratio2)
print "The ratio of 3rd class passenger survived: " + str(ratio3)

## Step 4: check the validity of the age model on cv set
cv_survived = []
for cv_row in crossVSet:
##    if cv_row[2] == '1' or cv_row[2] == '2':
    if cv_row[2] == '1':
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

print "The accuracy of the Pclass model on cv set is: " + str(accuracy)
print "The ratio of the survived passengers in cv set is: " + \
      str(np.sum(crossVSet[0::,1].astype(np.float)) / \
          np.size(crossVSet[0::,1]))
