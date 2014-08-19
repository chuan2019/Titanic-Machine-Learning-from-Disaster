import csv                  as csv
import Tkinter              as tk
import tkFileDialog         as fd
import matplotlib.pyplot    as plt; plt.rcdefaults()
import numpy                as np
import matplotlib.pyplot    as plt

## Step 1: loading .csv training dataset
#root = tk.Tk()
#root.withdraw()
#file_path = fd.askopenfilename()
file_path='../data/csv/train.csv'
rawData = csv.reader(open(file_path,'rb'))
colnames = rawData.next()
## 0: 'PassengerId', 1: 'Survived', 2: 'Pclass', 3: 'Name',
## 4: 'Sex', 5: 'Age', 6: 'SibSp', 7: 'Parch', 8: 'Ticket',
## 9: 'Fare', 10: 'Cabin', 11: 'Embarked'
data = []
for row in rawData:
    data.append(row)
data = np.array(data)

## Step 2: preprocessing data 1
# split the training data set into new training set and cross-validating set
trainingSetSize = int(len(data)*3/4)
trainingSet = data[0:trainingSetSize]
crossVSet = data[trainingSetSize:len(data)]
##trainingSet = data[-trainingSetSize::]
##crossVSet = data[0:-trainingSetSize]
# split the new training set into survived and perished sets
alive = trainingSet[0::,1] == '1'
dead  = trainingSet[0::,1] == '0'
survived = trainingSet[alive,0::]
perished = trainingSet[dead ,0::]

total = len(trainingSet)
ratio = float(len(survived)) / float(total)
print "____________________________________________________________________"
print "The overall survival rate is: " + str(ratio)

## Step 3: exploretary analysis - finding features
# gender distributions
women = survived[0::,4] == 'female'
women_survived = np.size(survived[women,1])
men_survived = np.size(survived[0::,1]) - women_survived

women = perished[0::,4] == 'female'
women_perished = np.size(perished[women,1])
men_perished = np.size(perished[0::,1]) - women_perished

women_total = women_survived + women_perished
women_ratio = float(women_survived)/float(women_total)
men_total = men_survived + men_perished
men_ratio = float(men_survived) / float(men_total)

print "____________________________________________________________________"
print "Women survived: " + str(women_survived) + \
      ", ratio: " + str(women_ratio)
print "Women perished: " + str(women_perished) + \
      ", ratio: " + str(float(women_perished)/float(women_total))
print "Men survived: " + str(men_survived) + \
      ", ratio: " + str(men_ratio)
print "Men perished: " + str(men_perished) + \
      ", ratio: " + str(float(men_perished) / float(men_total))

# age distributions
ageCutoff = 15

haveAge = trainingSet[0::,5] != ''
ageData = trainingSet[haveAge,5].astype(np.float)
# Omitting NAs in age field
haveAge = survived[0::,5] != ''
ageData_survived = survived[haveAge,5].astype(np.float)
haveAge = perished[0::,5] != ''
ageData_perished = perished[haveAge,5].astype(np.float)

teen1 = ageData_survived <= ageCutoff
teen0 = ageData_perished <= ageCutoff

teen_survived = np.size(ageData_survived[teen1])
adult_survived = np.size(ageData_survived[0::]) - teen_survived
teen_perished = np.size(ageData_perished[teen0])
adult_perished = np.size(ageData_perished[0::])

teen = ageData[0::] <= ageCutoff
teen_total = np.size(ageData[teen])
adult_total = np.size(ageData[0::]) - teen_total

teen_ratio = float(teen_survived) / float(teen_total)
adult_ratio = float(adult_survived) / float(adult_total)

print "____________________________________________________________________"
print "The ratio of survival rate of teenage (Age <= " + str(ageCutoff) + "): " + str(teen_ratio)
print "The ratio of survival rate of adult (Age > " + str(ageCutoff) + "): " + str(adult_ratio)

# Pclass distribution
pClass1 = trainingSet[0::,2] == '1'
pcData1 = trainingSet[pClass1,0::]
pClass2 = trainingSet[0::,2] == '2'
pcData2 = trainingSet[pClass2,0::]
pClass3 = trainingSet[0::,2] == '3'
pcData3 = trainingSet[pClass3,0::]

pcRatio1 = np.sum(pcData1[0::,1].astype(np.float)) / np.size(pcData1[0::,1])
pcRatio2 = np.sum(pcData2[0::,1].astype(np.float)) / np.size(pcData2[0::,1])
pcRatio3 = np.sum(pcData3[0::,1].astype(np.float)) / np.size(pcData3[0::,1])

print "____________________________________________________________________"
print "The ratio of survival rate of 1st class passenger: " + str(pcRatio1)
print "The ratio of survival rate of 2nd class passenger: " + str(pcRatio2)
print "The ratio of survival rate of 3rd class passenger: " + str(pcRatio3)

## Step 4: verify the predictions on cv set
# Below is the list of the column names
# 0: 'PassengerId', 1: 'Survived', 2: 'Pclass', 3: 'Name',
# 4: 'Sex', 5: 'Age', 6: 'SibSp', 7: 'Parch', 8: 'Ticket',
# 9: 'Fare', 10: 'Cabin', 11: 'Embarked'
cv_survived_predict = []
beta1 = 0; beta2 = 0; beta3 = 0
for row in crossVSet:
    if row[4] == 'female':
        beta1 = women_ratio
    else:
        beta1 = men_ratio

    if row[5] == '':
        beta2 = ratio
    elif float(row[5]) <= ageCutoff:
        beta2 = teen_ratio
    else:
        beta2 = adult_ratio

    if row[2] == '1':
        beta3 = pcRatio1
    elif row[2] == '2':
        beta3 = pcRatio2
    else:
        beta3 = pcRatio3

##    average_score = beta1
    average_score = (beta1 + beta2 + beta3)/3

    if average_score >= 0.5:
        cv_survived_predict.append([row[0],row[1],'1', \
                                    beta1,beta2,beta3,average_score])
    else:
        cv_survived_predict.append([row[0],row[1],'0', \
                                    beta1,beta2,beta3,average_score])

cv_survived_predict = np.array(cv_survived_predict)

correct = 0
for cv_row in cv_survived_predict:
    if cv_row[1] == cv_row[2]:
        correct += 1
total = len(cv_survived_predict)
accuracy = float(correct) / float(total)

print "===================================================================="
print "The accuracy of the gap model #1 on cv set is: " + str(accuracy)

print "Now give a deep breath, and try test set, and submit for my first time!"
file_path='../data/csv/test.csv'
test_file = open(file_path,'rb')
testData = csv.reader(test_file)
testColnames = testData.next()
## 0: 'PassengerId', 1: 'Pclass', 2: 'Name', 3: 'Sex',
## 4: 'Age', 5: 'SibSp', 6: 'Parch', 7: 'Ticket',
## 8: 'Fare', 9: 'Cabin', 10: 'Embarked'

# Also open a new file for outputing predictions
predictions_file = open("gapModel.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])

test_survived_predict = []
beta1 = 0; beta2 = 0; beta3 = 0
for row in testData:
    if row[3] == 'female':
        beta1 = women_ratio
    else:
        beta1 = men_ratio

    if row[4] == '':
        beta2 = ratio
    elif float(row[5]) <= ageCutoff:
        beta2 = teen_ratio
    else:
        beta2 = adult_ratio

    if row[1] == '1':
        beta3 = pcRatio1
    elif row[1] == '2':
        beta3 = pcRatio2
    else:
        beta3 = pcRatio3

    average_score = (beta1 + beta2 + beta3)/3

    if average_score >= 0.5:
        predictions_file_object.writerow([row[0], '1'])
    else:
        predictions_file_object.writerow([row[0], '0'])

# Close out the files
test_file.close()
predictions_file.close()


