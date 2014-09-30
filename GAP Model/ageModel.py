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
file_path = '../data/train.csv'
rawData = csv.reader(open(file_path,'rb'))
colnames = rawData.next()
## 0: 'PassengerId', 1: 'Survived', 2: 'Pclass', 3: 'Name',
## 4: 'Sex', 5: 'Age', 6: 'SibSp', 7: 'Parch', 8: 'Ticket',
## 9: 'Fare', 10: 'Cabin', 11: 'Embarked'
data = []
for row in rawData:
    data.append(row)
data = np.array(data)

## Omitting NAs in age field
haveAge = data[0::,5] != ''
ageData = data[haveAge,0::]
data = ageData

## Step 2: split the training dataset into two parts: training set and cv set
trainingSetSize = int(len(data)*3/4)
trainingSet = data[0:trainingSetSize]
crossVSet = data[trainingSetSize:len(data)]

ageCutoff = 15

## Step 3: applying the age model to check if age is a good feature
teen_survived = np.sum(trainingSet[trainingSet[0::,5].astype(np.float) <= ageCutoff,\
                                    1].astype(np.float))
teen_onboard = np.size(trainingSet[trainingSet[0::,5].astype(np.float) <= ageCutoff,\
                                    1].astype(np.float))
adult_survived = np.sum(trainingSet[trainingSet[0::,5].astype(np.float) > ageCutoff,\
                                    1].astype(np.float))
adult_onboard = np.size(trainingSet[trainingSet[0::,5].astype(np.float) > ageCutoff,\
                                    1].astype(np.float))

teen_ratio = teen_survived / teen_onboard
adult_ratio = adult_survived / adult_onboard

print "The ratio of teenages survived: " + str(teen_ratio)
print "The ratio of adults survived: " + str(adult_ratio)

## Step 4: check the validity of the age model on cv set
cv_survived = []
for cv_row in crossVSet:
    if cv_row[5].astype(np.float) <= ageCutoff:
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

print "The accuracy of the age model on cv set is: " + str(accuracy)

## Step 5: plot data
plt.figure(1)
plt.subplot(3,1,1)
ageDataAll = trainingSet[0::,5].astype(np.float)
n, bins, patches = plt.hist(ageDataAll, 50, normed=0, facecolor='g', \
                            alpha=0.75, label='All')
ageDataSurvived = trainingSet[trainingSet[0::,1] == '1',5].astype(np.float)
plt.hist(ageDataSurvived, bins, normed=0, facecolor='r', \
                            alpha=0.75, label='Survived')
plt.ylabel('Count')
plt.title('Age Model (Accuracy:%.3f'%accuracy +')')
plt.legend(loc='upper right')
plt.axis([0, 80, 0, 40])
plt.grid(True)

plt.subplot(3,1,2)
ageDataAllF = trainingSet[trainingSet[0::,4] == 'female',5].astype(np.float)
n, bins, patches = plt.hist(ageDataAllF, 50, normed=0, facecolor='g', \
                            alpha=0.75, label='Female All')
ageDataSurvivedF = trainingSet[(trainingSet[0::,1] == '1') \
                               & (trainingSet[0::,4] == 'female'),5].astype(np.float)
plt.hist(ageDataSurvivedF, bins, normed=0, facecolor='r', \
                            alpha=0.75, label='Female Survived')
plt.ylabel('Count')
##plt.title('Age Distribution (Female)')
plt.legend(loc='upper right')
plt.axis([0, 80, 0, 40])
plt.grid(True)

plt.subplot(3,1,3)
ageDataAllM = trainingSet[trainingSet[0::,4] == 'male',5].astype(np.float)
n, bins, patches = plt.hist(ageDataAllM, 50, normed=0, facecolor='g', \
                            alpha=0.75, label='Male All')
ageDataSurvivedM = trainingSet[(trainingSet[0::,1] == '1') \
                               & (trainingSet[0::,4] == 'male'),5].astype(np.float)
plt.hist(ageDataSurvivedM, bins, normed=0, facecolor='r', \
                            alpha=0.75, label='Male Survived')
plt.xlabel('Age')
plt.ylabel('Count')
##plt.title('Age Distribution (Male)')
plt.legend(loc='upper right')
plt.axis([0, 80, 0, 40])
plt.grid(True)

plt.savefig('AgeDistribution')
plt.show()
