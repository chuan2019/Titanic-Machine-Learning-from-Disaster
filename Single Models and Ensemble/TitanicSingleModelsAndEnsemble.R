# setwd("C:/Users/Chuan/My Work/Kaggle Competitions/Titanic_Machine Learning from Disaster/My R Code/Code")

rm(list=ls())
cat("\014")

#### Preprocessing data ####
data.raw <- read.csv("../data/train.csv",na.strings="",
                     colClasses=c('integer','factor','factor','character',
                                  'factor','numeric','factor','factor',
                                  'character','numeric','factor','factor'))
dim(data.raw)
# [1] 891  12
names(data.raw)
# [1] "PassengerId" "Survived"    "Pclass"      "Name"        "Sex"         "Age"        
# [7] "SibSp"       "Parch"       "Ticket"      "Fare"        "Cabin"       "Embarked"   
## Since the Id, Name, Ticket, and Cabin are irrelevant information, I am going to ignore these variables. Therefore, I am going to use the following features for training.
data <- data.raw[,c(2,3,5:8,10,12)]
names(data)
# [1] "Survived" "Pclass"   "Sex"      "Age"      "SibSp"    "Parch"    "Fare"     "Embarked"

data <- na.omit(data)
dim(data)
# [1] 712   8

#### Preparing for Training ####
library(caret)
set.seed(5425)
InTrain <- createDataPartition(data$Age,p=0.7,list=FALSE)
data.train <- data[InTrain,]
dim(data.train)
# [1] 501   8
data.cv <- data[-InTrain,]
dim(data.cv)
# [1] 211   8

#### Training Model 0: Logistic Regression ####
set.seed(1521)
modFit.logit <- glm(Survived ~ ., data=data.train, family=binomial(logit))
summary(modFit.logit)
# Call:
#     glm(formula = Survived ~ ., family = binomial(logit), data = data.train)
# 
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -3.2370  -0.5521  -0.3038   0.5433   2.6410  
# 
# Coefficients:
#                   Estimate Std. Error z value Pr(>|z|)    
#     (Intercept)  5.248e+00  7.458e-01   7.037 1.97e-12 ***
#     Pclass2     -1.816e+00  4.350e-01  -4.174 2.99e-05 ***
#     Pclass3     -3.057e+00  4.717e-01  -6.480 9.16e-11 ***
#     Sexmale     -2.820e+00  2.896e-01  -9.739  < 2e-16 ***
#     Age         -5.043e-02  1.123e-02  -4.489 7.15e-06 ***
#     SibSp1      -1.052e-01  3.116e-01  -0.338   0.7355    
#     SibSp2      -1.088e+00  8.274e-01  -1.315   0.1885    
#     SibSp3      -2.253e+00  9.271e-01  -2.430   0.0151 *  
#     SibSp4      -2.171e+00  1.007e+00  -2.156   0.0311 *  
#     SibSp5      -1.589e+01  7.420e+02  -0.021   0.9829    
#     Parch1       5.224e-01  3.827e-01   1.365   0.1722    
#     Parch2       6.623e-01  5.137e-01   1.289   0.1973    
#     Parch3       5.942e-01  1.169e+00   0.508   0.6113    
#     Parch4      -1.510e+01  9.676e+02  -0.016   0.9875    
#     Parch5      -5.877e-01  1.198e+00  -0.490   0.6239    
#     Fare         9.646e-04  4.055e-03   0.238   0.8120    
#     EmbarkedQ   -1.100e+00  7.741e-01  -1.421   0.1555    
#     EmbarkedS   -6.167e-01  3.538e-01  -1.743   0.0813 .  
# ---
#     Signif. codes:  0 ?**?0.001 ?*?0.01 ??0.05 ??0.1 ??1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 677.91  on 500  degrees of freedom
# Residual deviance: 396.37  on 483  degrees of freedom
# AIC: 432.37
# 
# Number of Fisher Scoring iterations: 14

modPred.logit <- predict(modFit.logit, newdata=data.cv, type="response")
confusionMatrix(table(modPred.logit,data.cv$Survived))
# Confusion Matrix and Statistics
# 
# modPred0   0   1
#        0 107  29
#        1  21  54
# 
#                Accuracy : 0.763           
#                  95% CI : (0.6998, 0.8187)
#     No Information Rate : 0.6066          
#     P-Value [Acc > NIR] : 1.119e-06       
# 
#                   Kappa : 0.4949          
#  Mcnemar's Test P-Value : 0.3222          
#                                           
#             Sensitivity : 0.8359          
#             Specificity : 0.6506          
#          Pos Pred Value : 0.7868          
#          Neg Pred Value : 0.7200          
#              Prevalence : 0.6066          
#          Detection Rate : 0.5071          
#    Detection Prevalence : 0.6445          
#       Balanced Accuracy : 0.7433          
#                                           
#        'Positive' Class : 0     
## This shows that Logistic model is not indeed a good choice

#### Training Model 1: Decision Tree Model ####
library(rpart)
modFit.rpart <- rpart(Survived ~ ., data=data.train)
library(rattle)
fancyRpartPlot(modFit.rpart)

modPred.rpart <- predict(modFit.rpart, newdata=data.cv, type='class')
confusionMatrix(table(modPred.rpart,data.cv$Survived))
# Confusion Matrix and Statistics
# 
# modPred.rpart   0   1
#             0 107  29
#             1  21  54
# 
#                Accuracy : 0.763           
#                  95% CI : (0.6998, 0.8187)
#     No Information Rate : 0.6066          
#     P-Value [Acc > NIR] : 1.119e-06       
# 
#                   Kappa : 0.4949          
#  Mcnemar's Test P-Value : 0.3222          
#                                           
#             Sensitivity : 0.8359          
#             Specificity : 0.6506          
#          Pos Pred Value : 0.7868          
#          Neg Pred Value : 0.7200          
#              Prevalence : 0.6066          
#          Detection Rate : 0.5071          
#    Detection Prevalence : 0.6445          
#       Balanced Accuracy : 0.7433          
#                                           
#        'Positive' Class : 0   
## Clearly, this decision tree is not a good model either

#### Training Model 2: Random Forest Model ####
library(randomForest)
set.seed(2609)
modFit.rf <- randomForest(Survived ~ ., data=data.train, ntree=20)
modPred.rf <- predict(modFit.rf, newdata=data.cv, type="class")
confusionMatrix(table(modPred.rf,data.cv$Survived))
# Confusion Matrix and Statistics
# 
# modPred.rf   0   1
#          0 105  25
#          1  23  58
# 
#                Accuracy : 0.7725          
#                  95% CI : (0.7099, 0.8272)
#     No Information Rate : 0.6066          
#     P-Value [Acc > NIR] : 2.365e-07       
# 
#                   Kappa : 0.5213          
#  Mcnemar's Test P-Value : 0.8852          
#                                           
#             Sensitivity : 0.8203          
#             Specificity : 0.6988          
#          Pos Pred Value : 0.8077          
#          Neg Pred Value : 0.7160          
#              Prevalence : 0.6066          
#          Detection Rate : 0.4976          
#    Detection Prevalence : 0.6161          
#       Balanced Accuracy : 0.7596          
#                                           
#        'Positive' Class : 0     
## This shows that random forest model is even worse!

#### Training Model 3: K-Nearest Neighbor Model ####
set.seed(5425)
modFit.knn3 <- knn3(Survived ~ ., data=data.train, k=20)
modPred.knn3 <- predict(modFit.knn3, newdata=data.cv, type='class')
confusionMatrix(table(modPred.knn3,data.cv$Survived))
# Confusion Matrix and Statistics
# 
# 
# modPred.knn3  0  1
#            0 94 42
#            1 34 41
# 
#               Accuracy : 0.6398          
#                 95% CI : (0.5711, 0.7046)
#    No Information Rate : 0.6066          
#    P-Value [Acc > NIR] : 0.180           
# 
#                  Kappa : 0.2323          
# Mcnemar's Test P-Value : 0.422           
# 
#            Sensitivity : 0.7344          
#            Specificity : 0.4940          
#         Pos Pred Value : 0.6912          
#         Neg Pred Value : 0.5467          
#             Prevalence : 0.6066          
#         Detection Rate : 0.4455          
#   Detection Prevalence : 0.6445          
#      Balanced Accuracy : 0.6142          
# 
# 'Positive' Class : 0  

#### Training Model 4: Linear Discriminant Model ####
# modFit.lda <- lda(Survived ~ ., data=data.train)
data.train.lda <- data.train
data.train.lda$Pclass <- as.numeric(data.train.lda$Pclass)
data.train.lda$SibSp <- as.numeric(data.train.lda$SibSp)
data.train.lda$Parch <- as.numeric(data.train.lda$Parch)
modFit.lda <- lda(Survived ~ ., data=data.train.lda)
# Call:
#     lda(Survived ~ ., data = data.train.lda)
# 
# Prior probabilities of groups:
#     0         1 
# 0.5908184 0.4091816 
# 
# Group means:
#     Pclass   Sexmale      Age    SibSp    Parch     Fare  EmbarkedQ EmbarkedS
# 0 2.527027 0.8378378 30.61149 1.506757 1.347973 21.07881 0.05743243 0.8378378
# 1 1.853659 0.3024390 28.01912 1.507317 1.585366 51.81571 0.01951220 0.6975610
# 
# Coefficients of linear discriminants:
#     LD1
# Pclass    -0.9276080285
# Sexmale   -1.9410113047
# Age       -0.0286012906
# SibSp     -0.2287705153
# Parch      0.0442926628
# Fare      -0.0000381142
# EmbarkedQ -0.6375875489
# EmbarkedS -0.4693264233
data.cv.lda <- data.cv
data.cv.lda$Pclass <- as.numeric(data.cv.lda$Pclass)
data.cv.lda$SibSp <- as.numeric(data.cv.lda$SibSp)
data.cv.lda$Parch <- as.numeric(data.cv.lda$Parch)
modPred.lda <- predict(modFit.lda, newdata=data.cv.lda, method='predictive')
confusionMatrix(table(modPred.lda$class,data.cv.lda$Survived))
# Confusion Matrix and Statistics
# 
# modPred.lda  0   1
#          0 107  28
#          1  21  55
# 
#                Accuracy : 0.7678         
#                  95% CI : (0.7049, 0.823)
#     No Information Rate : 0.6066         
#     P-Value [Acc > NIR] : 5.211e-07      
# 
#                   Kappa : 0.5061         
#  Mcnemar's Test P-Value : 0.3914         
#                                          
#             Sensitivity : 0.8359         
#             Specificity : 0.6627         
#          Pos Pred Value : 0.7926         
#          Neg Pred Value : 0.7237         
#              Prevalence : 0.6066         
#          Detection Rate : 0.5071         
#    Detection Prevalence : 0.6398         
#       Balanced Accuracy : 0.7493         
#                                          
#        'Positive' Class : 0             

#### Training Model 5: Support Vector Machine Model ####
modFit.svm <- svm(Survived ~ ., data=data.train)
modPred.svm <- predict(modFit.svm, newdata=data.cv)
confusionMatrix(table(modPred.svm,data.cv$Survived))
# Confusion Matrix and Statistics
# 
# modPred.svm   0   1
#           0 109  29
#           1  19  54
# 
#               Accuracy : 0.7725          
#                 95% CI : (0.7099, 0.8272)
#    No Information Rate : 0.6066          
#    P-Value [Acc > NIR] : 2.365e-07       
# 
#                  Kappa : 0.513           
# Mcnemar's Test P-Value : 0.1939          
# 
#            Sensitivity : 0.8516          
#            Specificity : 0.6506          
#         Pos Pred Value : 0.7899          
#         Neg Pred Value : 0.7397          
#             Prevalence : 0.6066          
#         Detection Rate : 0.5166          
#   Detection Prevalence : 0.6540          
#      Balanced Accuracy : 0.7511          
# 
#       'Positive' Class : 0  

#### Aggregating the six models using random forest model ####
modPred.logit <- as.factor(modPred.logit) # Accuracy : 0.763 
modPred.rpart <- as.factor(modPred.rpart) # Accuracy : 0.763
modPred.rf    <- as.factor(modPred.rf)    # Accuracy : 0.7725
modPred.knn3  <- as.factor(modPred.knn3)  # Accuracy : 0.6398
modPred.lda   <- as.factor(modPred.lda$class) # Accuracy : 0.7678
modPred.svm   <- as.factor(modPred.svm)   # Accuracy : 0.7725
modPreds <- data.frame(logit=modPred.logit, rpart=modPred.rpart,
                       rf   =modPred.rf,    knn3 =modPred.knn3,
                       lda  =modPred.lda,   svm  =modPred.svm,
                       Survived=data.cv$Survived)
library(randomForest)
modFit.Agg  <- randomForest(Survived ~ ., data=modPreds)
# Call:
#     randomForest(formula = Survived ~ ., data = modPreds) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 2
# 
# OOB estimate of  error rate: 24.17%
# Confusion matrix:
#     0  1 class.error
# 0 109 19   0.1484375
# 1  32 51   0.3855422

#### Aggregating the six models using majority vote model ####
modPred.logit <- as.numeric(modPred.logit) # Accuracy : 0.763 
modPred.rpart <- as.numeric(modPred.rpart) # Accuracy : 0.763
modPred.rf    <- as.numeric(modPred.rf)    # Accuracy : 0.7725
modPred.knn3  <- as.numeric(modPred.knn3)  # Accuracy : 0.6398
modPred.lda   <- as.numeric(modPred.lda)   # Accuracy : 0.7678
modPred.svm   <- as.numeric(modPred.svm)   # Accuracy : 0.7725

modPred.majv  <- modPred.logit + modPred.rpart + modPred.rf + 
    modPred.knn3 + modPred.lda + modPred.svm

modPreds      <- ifelse(modPred.majv > 9, 1, 0)
confusionMatrix(table(modPreds,data.cv$Survived))
# Confusion Matrix and Statistics
# 
# modPreds   0   1
#        0 109  31
#        1  19  52
# 
#                Accuracy : 0.763           
#                  95% CI : (0.6998, 0.8187)
#     No Information Rate : 0.6066          
#     P-Value [Acc > NIR] : 1.119e-06       
# 
#                   Kappa : 0.4905          
#  Mcnemar's Test P-Value : 0.1198          
#                                           
#             Sensitivity : 0.8516          
#             Specificity : 0.6265          
#          Pos Pred Value : 0.7786          
#          Neg Pred Value : 0.7324          
#              Prevalence : 0.6066          
#          Detection Rate : 0.5166          
#    Detection Prevalence : 0.6635          
#       Balanced Accuracy : 0.7390          
#                                           
#        'Positive' Class : 0  
