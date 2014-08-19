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

## Step 2: splitting data into survived and perished groups
alive = data[0::,1] == '1'
dead  = data[0::,1] == '0'
survived = data[alive,0::]
perished = data[dead ,0::]
total = len(data)

## Step 3: explore gender distributions
women = survived[0::,4] == 'female'
women_survived = np.size(survived[women,1])
men_survived = np.size(survived[0::,1]) - women_survived

women = perished[0::,4] == 'female'
women_perished = np.size(perished[women,1])
men_perished = np.size(perished[0::,1]) - women_perished

women_total = women_survived + women_perished
men_total = men_survived + men_perished

women_ratio = float(women_survived)/float(women_total)
men_ratio = float(men_survived) / float(men_total)

print "Women survived: " + str(women_survived) + \
      ", ratio: " + str(women_ratio)
print "Women perished: " + str(women_perished) + \
      ", ratio: " + str(float(women_perished)/float(women_total))
print "Men survived: " + str(men_survived) + \
      ", ratio: " + str(men_ratio)
print "Men perished: " + str(men_perished) + \
      ", ratio: " + str(float(men_perished) / float(men_total))

## Step 4: explore age distributions
ageCutoff = 15

haveAge = data[0::,5] != ''
ageData = data[haveAge,5].astype(np.float)
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

print "The ratio of survival rate of teenage (Age <= " + str(ageCutoff) + "): " + str(teen_ratio)
print "The ratio of survival rate of adult (Age > " + str(ageCutoff) + "): " + str(adult_ratio)

plt.figure(1)
plt.subplot(211)
n, bins, patches = plt.hist(ageData, 50, normed=0, facecolor='g', \
                            alpha=0.75, label='All')
plt.hist(ageData_survived, bins, normed=0, facecolor='b', \
         alpha=0.75, label='Survived')
plt.ylabel('Count')
plt.title('Age Distribution (Survived)')
plt.axis([0, 80, 0, 60])
plt.grid(True)

plt.subplot(212)
n, bins, patches = plt.hist(ageData, 50, normed=0, facecolor='g', \
                            alpha=0.75, label='All')
plt.hist(ageData_perished, bins, normed=0, facecolor='r', \
         alpha=0.75, label='Perished')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution (Perished)')
plt.axis([0, 80, 0, 60])
plt.grid(True)

plt.savefig('figure_1-AgeDistribution')

## Step 5: explore Pclass distributions
pClass1 = data[0::,2] == '1'
pcData1 = data[pClass1,0::]
pClass2 = data[0::,2] == '2'
pcData2 = data[pClass2,0::]
pClass3 = data[0::,2] == '3'
pcData3 = data[pClass3,0::]

pcRatio1 = np.sum(pcData1[0::,1].astype(np.float)) / np.size(pcData1[0::,1])
pcRatio2 = np.sum(pcData2[0::,1].astype(np.float)) / np.size(pcData2[0::,1])
pcRatio3 = np.sum(pcData3[0::,1].astype(np.float)) / np.size(pcData3[0::,1])

print "The ratio of survival rate of 1st class passenger: " + str(pcRatio1)
print "The ratio of survival rate of 2nd class passenger: " + str(pcRatio2)
print "The ratio of survival rate of 3rd class passenger: " + str(pcRatio3)

## Step 6: explore Fare price distributions (see FP model for more details)
# 0: 'PassengerId', 1: 'Survived', 2: 'Pclass', 3: 'Name',
# 4: 'Sex', 5: 'Age', 6: 'SibSp', 7: 'Parch', 8: 'Ticket',
# 9: 'Fare', 10: 'Cabin', 11: 'Embarked'


## Plot the ratios
plt.figure(2)
plt.title('Survival Rate Distribution')

plt.subplot(1,3,1)
genderSurvivalRate = [women_ratio, men_ratio]
gender = ["women","men"]
x_pos = np.arange(len(gender))
plt.bar(x_pos, genderSurvivalRate, align='center', alpha=0.4, facecolor='r')
plt.xticks(x_pos, gender)
plt.xlabel('gender')
plt.title('Gender Model')
plt.ylim([0, 1])
plt.grid(True)

plt.subplot(1,3,2)
ageSurvivalRate = [teen_ratio,adult_ratio]
ageS = ["<=%d" % ageCutoff, ">%d" % ageCutoff]
x_pos = np.arange(len(ageS))
plt.bar(x_pos, ageSurvivalRate, align='center', alpha=0.4, facecolor='b')
plt.xticks(x_pos, ageS)
plt.xlabel('age')
plt.title('Age Model')
plt.ylim([0, 1])
plt.grid(True)

plt.subplot(1,3,3)
pcSurvivalRate = [pcRatio1,pcRatio2,pcRatio3]
pcS = ["PC1","PC2","PC3"]
x_pos = np.arange(len(pcS))
plt.bar(x_pos, pcSurvivalRate, align='center', alpha=0.4, facecolor='g')
plt.xticks(x_pos, pcS)
plt.xlabel('passenger class')
plt.title('Pclass Model')
plt.ylim([0, 1])
plt.grid(True)

plt.savefig('figure_2-SurvivalRate')

plt.show()
