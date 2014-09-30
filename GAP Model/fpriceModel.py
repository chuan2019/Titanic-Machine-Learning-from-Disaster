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

## Omitting NAs in Pclass field
haveFare = data[0::,9] != ''
fpData = data[haveFare,0::]
data = fpData
averageFP = np.mean(data[0::,9].astype(np.float))
MaxFP = 50
BinNumFP = 10
BinSizeFP = MaxFP / BinNumFP
FPScoreA = np.zeros([BinNumFP],float)
FPScoreF = np.zeros([BinNumFP],float)
FPScoreM = np.zeros([BinNumFP],float)
data[data[0::,9].astype(np.float) > MaxFP, 9] = MaxFP - 1

## Step 2: split the training dataset into two parts: training set and cv set
trainingSetSize = int(len(data)*3/4)
trainingSet = data[0:trainingSetSize]
crossVSet = data[trainingSetSize:len(data)]

## Step 3: get the FPScore, and use it to predict the survival of passengers
for i in xrange(BinNumFP):
    FPSurvivedA = trainingSet[(trainingSet[0::,9].astype(np.float) \
                              >=i*BinSizeFP) & \
                             (trainingSet[0::,9].astype(np.float) \
                              <(i+1)*BinSizeFP),1].astype(np.float)
    FPSurvivedA = np.sum(FPSurvivedA)
    FPTotalA = float(len(trainingSet[(trainingSet[0::,9].astype(np.float) \
                              >=i*BinSizeFP) & \
                             (trainingSet[0::,9].astype(np.float) \
                              <(i+1)*BinSizeFP),1]))
    if FPTotalA == 0:
        FPScoreA[i] = 0
    else:
        FPScoreA[i] = FPSurvivedA / FPTotalA
    print "FPScoreA["+str(i)+"] = "+str(FPSurvivedA)+"/"+str(FPTotalA)+\
          " = "+str(FPScoreA[i])

    FPSurvivedF = trainingSet[(trainingSet[0::,9].astype(np.float) \
                              >=i*BinSizeFP) & \
                             (trainingSet[0::,9].astype(np.float) \
                              <(i+1)*BinSizeFP) & \
                              (trainingSet[0::,4]=='female'),1].astype(np.float)
    FPSurvivedF = np.sum(FPSurvivedF)
    FPTotalF = float(len(trainingSet[(trainingSet[0::,9].astype(np.float) \
                              >=i*BinSizeFP) & \
                             (trainingSet[0::,9].astype(np.float) \
                              <(i+1)*BinSizeFP) & \
                              (trainingSet[0::,4]=='female'),1]))
    if FPTotalF == 0:
        FPScoreF[i] = 0
    else:
        FPScoreF[i] = FPSurvivedF / FPTotalF
    print "FPScoreF["+str(i)+"] = "+str(FPSurvivedF)+"/"+str(FPTotalF)+\
          " = "+str(FPScoreF[i])

    FPSurvivedM = trainingSet[(trainingSet[0::,9].astype(np.float) \
                              >=i*BinSizeFP) & \
                             (trainingSet[0::,9].astype(np.float) \
                              <(i+1)*BinSizeFP) & \
                              (trainingSet[0::,4]=='male'),1].astype(np.float)
    FPSurvivedM = np.sum(FPSurvivedM)
    FPTotalM = float(len(trainingSet[(trainingSet[0::,9].astype(np.float) \
                              >=i*BinSizeFP) & \
                             (trainingSet[0::,9].astype(np.float) \
                              <(i+1)*BinSizeFP) & \
                              (trainingSet[0::,4]=='male'),1]))
    if FPTotalM == 0:
        FPScoreM[i] = 0
    else:
        FPScoreM[i] = FPSurvivedM / FPTotalM
    print "FPScoreM["+str(i)+"] = "+str(FPSurvivedM)+"/"+str(FPTotalM)+\
          " = "+str(FPScoreM[i])

## Step 4: check accuracy of the prediction on cv set
cv_survived = []
for cv_row in crossVSet:
    for i in xrange(BinNumFP):
        if cv_row[4] == 'female':
            if (float(cv_row[9])>=i*BinSizeFP) & \
               (float(cv_row[9])<(i+1)*BinSizeFP):
                if FPScoreF[i] > 0.5:
                    cv_survived.append([cv_row[0],cv_row[1],'1'])
                else:
                    cv_survived.append([cv_row[0],cv_row[1],'0'])
                break
        else:
            if (float(cv_row[9])>=i*BinSizeFP) & \
               (float(cv_row[9])<(i+1)*BinSizeFP):
                if FPScoreM[i] > 0.5:
                    cv_survived.append([cv_row[0],cv_row[1],'1'])
                else:
                    cv_survived.append([cv_row[0],cv_row[1],'0'])
                break
cv_survived = np.array(cv_survived)

correct = 0
for cv_row in cv_survived:
    if cv_row[1] == cv_row[2]:
        correct += 1
total = len(cv_survived)
accuracy = float(correct) / float(total)

print "The accuracy of the fp model on cv set is: " + str(accuracy)

## Step 5: plot fp data
plt.figure(1)

plt.subplot(3,1,1)
fpDataAll = trainingSet[0::,9].astype(np.float)
n, bins, patches = plt.hist(fpDataAll, 50, normed=0, facecolor='g', \
                            alpha=0.75, label='All')
fpDataSurvived = trainingSet[trainingSet[0::,1]=='1',9].astype(np.float)
plt.hist(fpDataSurvived, bins, normed=0, facecolor='r', \
         alpha=0.75, label='Survived')
plt.ylabel('Count')
plt.title('Fare Price Model (All)')
plt.legend(loc='upper right')
plt.axis([0, 50, 0, 170])
plt.grid(True)

plt.subplot(3,1,2)
fpDataAllF = trainingSet[trainingSet[0::,4]=='female',9].astype(np.float)
n, bins, patches = plt.hist(fpDataAllF, 50, normed=0, facecolor='g', \
                            alpha=0.75, label='Female All')
fpDataSurvivedF = trainingSet[(trainingSet[0::,1]=='1') \
                              & (trainingSet[0::,4]=='female'),9].astype(np.float)
plt.hist(fpDataSurvivedF, bins, normed=0, facecolor='r', \
         alpha=0.75, label='Female Survived')
plt.ylabel('Count')
plt.legend(loc='upper right')
plt.axis([0, 50, 0, 80])
plt.grid(True)

plt.subplot(3,1,3)
fpDataAllM = trainingSet[trainingSet[0::,4]=='male',9].astype(np.float)
n, bins, patches = plt.hist(fpDataAllM, 50, normed=0, facecolor='g', \
                            alpha=0.75, label='Male All')
fpDataSurvivedM = trainingSet[(trainingSet[0::,1]=='1') \
                              & (trainingSet[0::,4]=='male'),9].astype(np.float)
plt.hist(fpDataSurvivedM, bins, normed=0, facecolor='r', \
         alpha=0.75, label='Male Survived')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.legend(loc='upper right')
plt.axis([0, 50, 0, 130])
plt.grid(True)

plt.savefig('FPSurvivalRate')


plt.figure(2)

fpS = []
for i in xrange(BinNumFP):
    fpS.append(("%d" % int(float(i+0.5)*BinSizeFP)))
x_pos = np.arange(len(fpS))

plt.subplot(2,2,1)
plt.bar(x_pos, FPScoreA, align='center', alpha=0.4, facecolor='g')
plt.xticks(x_pos, fpS)
plt.xlabel('Fare Price')
plt.title('Fare (All)')
plt.ylim([0, 1])
plt.grid(True)

plt.subplot(2,2,2)
plt.bar(x_pos, FPScoreF, align='center', alpha=0.4, facecolor='b')
plt.xticks(x_pos, fpS)
##plt.xlabel('Fare Price')
plt.title('Fare (Female)')
plt.ylim([0, 1])
plt.grid(True)

plt.subplot(2,2,4)
plt.bar(x_pos, FPScoreM, align='center', alpha=0.4, facecolor='r')
plt.xticks(x_pos, fpS)
plt.xlabel('Fare Price')
plt.title('Fare (Male)')
plt.ylim([0, 1])
plt.grid(True)

plt.savefig('FPScore')

plt.show()

