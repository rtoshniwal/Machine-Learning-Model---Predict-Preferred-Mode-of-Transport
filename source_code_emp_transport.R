#======================================================================= 
# 
#   Employee Mode of Transport Prediction 
# 
#======================================================================= 

library(dplyr)
library(mice)
library(ggplot2)
library(gridExtra)
library(corrplot)
library(ppcor)
library(caTools)
library(DMwR)
library(car)
library(class)
library(e1071)
library(gbm)
library(xgboost)
library(caret)
library(ipred)
library(rpart)
library(ROCR)
library(ineq)

#Set up the working directory
PATH = 'R:/Great Learning/8.Machine Learning/Project 5'
setwd(PATH)

#Import the dataset
empData = read.csv('Cars.csv', header = T)

## Descriptive Statistics ##
#Getting a compact structure of the data
str(empData)

#Checking missing values
sapply(empData,function(x) sum(is.na(x)))

#Impute missing values
#md.pattern(empData)
data2=empData
data2$MBA = as.factor(data2$MBA) 
tempData <- mice(data2,m=5,maxit=50,meth='logreg',seed=500)
tempData$imp$MBA

empData$MBA[which(is.na(empData$MBA))] = 0 # Since the 5th logreg iteration using mice gives MBA = 0

#Dataset Summary Continuous variables
summary(empData)

#Observations
# 1. Dependent variable 'Transport' is a ternary variable. 
#    To make binary predictions for the employee mode of transport i.e. to predict if employee uses car or not, make a new catagorical 
#    variable Car where, the value is 1 if employee uses car and 0 if employee uses other mode of transport.
# 2. All independent variables are int or numeric except Gender
# 3. One missing value in MBA.
# 4. Max value of Work.Exp,Salary and Distance much higher than the 3rd quartile - indicating a possibility of outliers.


#Dataset Summary Categorical variables

barplotfn = function(variable,variableNameString){
  ggplot(data = empData, aes(variable))+
    labs(x = variableNameString,y ='Count')+
    geom_bar(aes(fill = Transport),col = 'white',width=0.5)+
    scale_fill_manual(values = c('navy','green','skyblue'))+
    theme(aspect.ratio = 1)
}

table(empData$Gender)
barplotfn(empData$Gender,'Gender')
table(as.factor(empData$Engineer))
barplotfn(as.factor(empData$Engineer),'Engineer')
table(as.factor(empData$MBA))
barplotfn(as.factor(empData$MBA),'MBA')
table(as.factor(empData$license))
barplotfn(as.factor(empData$license),'License')


## Data Visualisation ##
#Univariate Analysis - Histogram and Boxplots of continuous variables

plot_histogram_n_barplot = function(variable, variableNameString){
  h = ggplot(data = empData, aes(variable))+
    labs(x = variableNameString,y ='Count')+
    geom_histogram(fill = 'green',col = 'white')
  b = ggplot(data = empData, aes('',variable))+ 
    geom_boxplot(outlier.colour = 'red',col = 'black')+
    labs(x = '',y = variableNameString)+ coord_flip()
  grid.arrange(h,b,ncol = 2)
}

plot_histogram_n_barplot(empData$Age,'Age')
plot_histogram_n_barplot(empData$Work.Exp,'Work Experience')
plot_histogram_n_barplot(empData$Salary,'Salary')
plot_histogram_n_barplot(empData$Distance,'Distance (Office - Home)')

#Bivariate Analysis

#Create a newbinary dependent factor variable using the ternary variable 'Transport'

table(empData$Transport)
empData$Car = as.factor(as.integer(empData$Transport == 'Car'))
table(empData$Car)

#Dependent variable vs continuous independent variables
bivariate_plots = function(variable,variableNameString){
  ggplot(data = empData, aes(as.factor(empData$Car),variable))+ 
    geom_boxplot(outlier.colour = 'blue',col = 'blue',fill='cyan')+
    labs(x = 'Car',y=variableNameString)
}
bivariate_plots(empData$Age,'Age')
bivariate_plots(empData$Work.Exp,'Work Experience')
bivariate_plots(empData$Salary,'Salary')
bivariate_plots(empData$Distance,'Distance (Office - Home)')


#Dependent variable vs categorical independent variables

table(empData$Car,empData$Gender)
table(empData$Car,empData$Engineer)
table(empData$Car,empData$MBA)
table(empData$Car,empData$license)

#Update Gender to Female
#For plotting out correlationplots, att the variables must be int/numeric
empData$Female = as.numeric(empData$Gender=='Female')
empDataUpdated = empData[,c(1,3:8,11,10)]

empDataUpdated$Car=as.numeric(as.character(empDataUpdated$Car))

#Correlation Plot
datamatrix = cor(empDataUpdated)
corrplot(datamatrix, method = 'number',type = 'upper' ,number.cex = 1.0)

#Correlation Significance using p-values of t-statistics
corsig = data.frame(pcor(empDataUpdated,method = 'pearson'))
corsig

#Data Preparation
table(empDataUpdated$Car)
length(which(empDataUpdated$Car==1))/nrow(empDataUpdated)
ggplot(data = empDataUpdated, aes(empDataUpdated$Car))+
  labs(x = 'Car',y ='Count')+
  geom_bar(fill='red',col = 'white',width=0.5)+
  scale_fill_manual(values = c('navy','green','skyblue'))+
  theme(aspect.ratio = 1)

#Split the data into train and validation set

set.seed(1000)
#Update all the factor variables fromnumericto factor
empDataUpdated$Car = as.factor(as.character(empDataUpdated$Car))
empDataUpdated$Engineer = as.factor(as.character(empDataUpdated$Engineer))
empDataUpdated$MBA = as.factor(as.character(empDataUpdated$MBA))
empDataUpdated$license = as.factor(as.character(empDataUpdated$license))
empDataUpdated$Female = as.factor(as.character(empDataUpdated$Female))
sample = sample.split(empDataUpdated$Car, SplitRatio = 0.8)
train = subset(empDataUpdated,sample == TRUE) #Development
val = subset(empDataUpdated,sample == FALSE) #Hold-out

#Check that the split created comparable samples
table(empDataUpdated$Car)
prop.table(table(empDataUpdated$Car))
table(train$Car)
prop.table(table(train$Car))
table(val$Car)
prop.table(table(val$Car))

#Balance Train Data using SMOTE
#The dependent varible shouldbe a factor.

class(train$Car)

balanced.train = SMOTE(Car ~.,train, perc.over = 600, k= 5, perc.under = 200)
table(balanced.train$Car)
ggplot(data = balanced.train, aes(balanced.train$Car))+
  labs(x = 'Car',y ='Count')+
  geom_bar(fill='red',col = 'white',width=0.5)+
  scale_fill_manual(values = c('navy','green','skyblue'))+
  theme(aspect.ratio = 1)

## ModelBuilding ##
#Logistic Regression
logit.train = balanced.train
str(logit.train)

LRmodel1=glm(Car~.,data=logit.train,family = 'binomial')
summary(LRmodel1)

#CheckVIF
vif(LRmodel1)

# VIF of Work.Exp ismax.Drop Work.Exp variable, build a new lr model and check the vif again
logit.train = subset(logit.train[,-c(4)])

LRmodel2 = glm(Car~.,data=logit.train,family = 'binomial')
summary(LRmodel2)

#The significance of a lot of variables has increased,check the VIF again
vif(LRmodel2)

# VIFs are within the acceptable range now. Drop  MBA because it is not significant.
# And check the AIC.If the AIC decreases, we must drop these variables from the final model.
logit.train = subset(logit.train[,-c(3)])

LRmodel3 = glm(Car~.,data=logit.train,family = 'binomial')
summary(LRmodel3)

# The AIC of LRmodel3 is less than that of LRmodel2. Hence, our final logistic regression model will be LRModel3

#Interpretting the logistic regression model
#Log Odds Ratio
a=coef(LRmodel3)
#Odds
b=exp(coef(LRmodel3))
#Probablity
c=exp(coef(LRmodel3))/(1+exp(coef(LRmodel3)))


#Model Acceptability
#Insample
pred = predict(LRmodel3,newdata = logit.train,type = 'response')
train_Car_pred = ifelse(pred >0.5,1,0)
train_Car_actual = logit.train$Car
tab.logit.train = table(train_Car_actual,train_Car_pred)
tab.logit.train

Accuracy.logit.train = sum(diag(tab.logit.train)/sum(tab.logit.train))
Accuracy.logit.train
Sensitivity.logit.train = tab.logit.train[2,2]/(tab.logit.train[2,1]+tab.logit.train[2,2])
Sensitivity.logit.train
Specificity.logit.train = tab.logit.train[1,1]/(tab.logit.train[1,1]+tab.logit.train[1,2])
Specificity.logit.train
Precision.logit.train = tab.logit.train[2,2]/(tab.logit.train[1,2]+tab.logit.train[2,2])
Precision.logit.train

auc.perf.logit=performance(prediction(train_Car_pred,train_Car_actual),"auc")
AUC.logit.train = attr(auc.perf.logit,"y.values")[[1]]
AUC.logit.train


#Validation Set
pred = predict(LRmodel3,newdata = val,type = 'response')
val_pred = ifelse(pred >0.5,1,0)
val_actual = val$Car
tab.logit.val = table(val_actual,val_pred)
tab.logit.val
Accuracy.logit.val = sum(diag(tab.logit.val)/sum(tab.logit.val))
Accuracy.logit.val
Sensitivity.logit.val = tab.logit.val[2,2]/(tab.logit.val[2,1]+tab.logit.val[2,2])
Sensitivity.logit.val
Specificity.logit.val = tab.logit.val[1,1]/(tab.logit.val[1,1]+tab.logit.val[1,2])
Specificity.logit.val
Precision.logit.val = tab.logit.val[2,2]/(tab.logit.val[1,2]+tab.logit.val[2,2])
Precision.logit.val

auc.perf.logit=performance(prediction(val_pred,val_actual),"auc")
AUC.logit.val = attr(auc.perf.logit,"y.values")[[1]]
AUC.logit.val

#********************************************************************************************************
#k Nearest Neighbour

#Normalise the variables
normalise = function(x){
  return ((x-min(x))/(max(x)-min(x)))
  }

knn.train = balanced.train
str(knn.train)
knn.train$Engineer = as.numeric(as.character(knn.train$Engineer))
knn.train$MBA = as.numeric(as.character(knn.train$MBA))
knn.train$Female = as.numeric(as.character(knn.train$Female))
knn.train$license = as.numeric(as.character(knn.train$license))
knn.train$Age = normalise(knn.train$Age)
knn.train$Work.Exp = normalise(knn.train$Work.Exp)
knn.train$Salary = normalise(knn.train$Salary)
knn.train$Distance = normalise(knn.train$Distance)

knn.val = val
knn.val$Engineer = as.numeric(as.character(knn.val$Engineer))
knn.val$MBA = as.numeric(as.character(knn.val$MBA))
knn.val$Female = as.numeric(as.character(knn.val$Female))
knn.val$license = as.numeric(as.character(knn.val$license))
knn.val$Age = normalise(knn.val$Age)
knn.val$Work.Exp = normalise(knn.val$Work.Exp)
knn.val$Salary = normalise(knn.val$Salary)
knn.val$Distance = normalise(knn.val$Distance)

#knn3
pred = knn(train = knn.train[,-1], test =knn.val[-1], knn.train$Car, k = 3) 
tab.knn.val = table(knn.val$Car, pred)
tab.knn.val

Accuracy.knn = sum(diag(tab.knn.val)/sum(tab.knn.val))
Accuracy.knn
Sensitivity.knn = tab.knn.val[2,2]/(tab.knn.val[2,1]+tab.knn.val[2,2])
Sensitivity.knn
Specificity.knn = tab.knn.val[1,1]/(tab.knn.val[1,1]+tab.knn.val[1,2])
Specificity.knn
Precision.knn = tab.knn.val[2,2]/(tab.knn.val[1,2]+tab.knn.val[2,2])
Precision.knn



auc.perf.kNN=performance(prediction(as.numeric(as.character(pred)),knn.val$Car),"auc")
AUC.knn = attr(auc.perf.kNN,"y.values")[[1]]
AUC.knn


#*********************************************************************************

#Naive Bayes Model
#Dependent variable should be factor
nb.train = balanced.train
str(nb.train)
nb.val = val

nbModel = naiveBayes(x = nb.train[,-1], y = nb.train$Car)

#Model Acceptability
#Insample
predNB.train = predict(nbModel, nb.train, type = "class")
tab.nb.train = table(nb.train$Car,predNB.train)
tab.nb.train

Accuracy.nb.train = sum(diag(tab.nb.train)/sum(tab.nb.train))
Accuracy.nb.train
Sensitivity.nb.train = tab.nb.train[2,2]/(tab.nb.train[2,1]+tab.nb.train[2,2])
Sensitivity.nb.train
Specificity.nb.train = tab.nb.train[1,1]/(tab.nb.train[1,1]+tab.nb.train[1,2])
Specificity.nb.train
Precision.nb.train = tab.nb.train[2,2]/(tab.nb.train[1,2]+tab.nb.train[2,2])
Precision.nb.train

auc.perf.nb=performance(prediction(as.numeric(as.character(predNB.train)),nb.train$Car),"auc")
AUC.nb.train = attr(auc.perf.nb,"y.values")[[1]]
AUC.nb.train

#On validation Data
predNB = predict(nbModel, nb.val, type = "class")
tab.nb.val = table(nb.val$Car, predNB)
tab.nb.val

Accuracy.nb.val = sum(diag(tab.nb.val)/sum(tab.nb.val))
Accuracy.nb.val
Sensitivity.nb.val = tab.nb.val[2,2]/(tab.nb.val[2,1]+tab.nb.val[2,2])
Sensitivity.nb.val
Specificity.nb.val = tab.nb.val[1,1]/(tab.nb.val[1,1]+tab.nb.val[1,2])
Specificity.nb.val
Precision.nb.val = tab.nb.val[2,2]/(tab.nb.val[1,2]+tab.nb.val[2,2])
Precision.nb.val

auc.perf.nb=performance(prediction(as.numeric(as.character(predNB)),nb.val$Car),"auc")
AUC.nb.val = attr(auc.perf.nb,"y.values")[[1]]
AUC.nb.val

#*******************************************************************************************************************
#Bagging
bagging.train = train
bagging.val= val
bagging.model = bagging(Car~.,
                        data = bagging.train,
                        control = rpart.control(maxdepth = 5, minsplit = 10))
#Insample performance
bagging.train$pred.Car = predict(bagging.model,bagging.train)
tab.bag.train = table(bagging.train$Car,bagging.train$pred.Car)
tab.bag.train
sum(diag(tab.bag.train)/sum(tab.bag.train))

Accuracy.bag.train = sum(diag(tab.bag.train)/sum(tab.bag.train))
Accuracy.bag.train
Sensitivity.bag.train = tab.bag.train[2,2]/(tab.bag.train[2,1]+tab.bag.train[2,2])
Sensitivity.bag.train
Specificity.bag.train = tab.bag.train[1,1]/(tab.bag.train[1,1]+tab.bag.train[1,2])
Specificity.bag.train
Precision.bag.train = tab.bag.train[2,2]/(tab.bag.train[1,2]+tab.bag.train[2,2])
Precision.bag.train

auc.perf.bag=performance(prediction(as.numeric(as.character(bagging.train$pred.Car)),bagging.train$Car),"auc")
AUC.bag.train = attr(auc.perf.bag,"y.values")[[1]]
AUC.bag.train


#Validation data performance
bagging.val$pred.Car = predict(bagging.model,bagging.val)
tab.bag.val = table(bagging.val$Car,bagging.val$pred.Car)
tab.bag.val

Accuracy.bag.val = sum(diag(tab.bag.val)/sum(tab.bag.val))
Accuracy.bag.val
Sensitivity.bag.val = tab.bag.val[2,2]/(tab.bag.val[2,1]+tab.bag.val[2,2])
Sensitivity.bag.val
Specificity.bag.val = tab.bag.val[1,1]/(tab.bag.val[1,1]+tab.bag.val[1,2])
Specificity.bag.val
Precision.bag.val = tab.bag.val[2,2]/(tab.bag.val[1,2]+tab.bag.val[2,2])
Precision.bag.val

auc.perf.bag=performance(prediction(as.numeric(as.character(bagging.val$pred.Car)),bagging.val$Car),"auc")
AUC.bag.val = attr(auc.perf.bag,"y.values")[[1]]
AUC.bag.val

#*******************************************************************************************************************
#Boosting

#All variables must be numeric for xgboost

boosting.train = train
boosting.train$Engineer = as.numeric(as.character(boosting.train$Engineer))
boosting.train$MBA = as.numeric(as.character(boosting.train$MBA))
boosting.train$license = as.numeric(as.character(boosting.train$license))
boosting.train$Female = as.numeric(as.character(boosting.train$Female))
boosting.train$Car = as.numeric(as.character(boosting.train$Car))

boosting.val = val
boosting.val$Engineer = as.numeric(as.character(boosting.val$Engineer))
boosting.val$MBA = as.numeric(as.character(boosting.val$MBA))
boosting.val$license = as.numeric(as.character(boosting.val$license))
boosting.val$Female = as.numeric(as.character(boosting.val$Female))
boosting.val$Car = as.numeric(as.character(boosting.val$Car))

#We need to separate training data and the dependent variable
#Xgboost works with matrices

boosting.train.features = as.matrix(boosting.train[,-9])
boosting.train.label = as.matrix(boosting.train[,9])
boosting.val.features = as.matrix(boosting.val[,-9])

xgbModel = xgboost(
  data = boosting.train.features,
  label = boosting.train.label,
  eta = 0.001,
  max_depth = 3,
  min_child_weight = 3,
  nrounds = 1000,
  nfold= 5,
  objective = 'binary:logistic',
  verbose = 0,
  early_topping_rounds=10
)

#Insample performance
boosting.train$pred.Car = predict(xgbModel,boosting.train.features)
tab.boost.train = table(boosting.train$Car,boosting.train$pred.Car>0.5)
tab.boost.train
sum(diag(tab.boost.train)/sum(tab.boost.train))

Accuracy.boost.train = sum(diag(tab.boost.train)/sum(tab.boost.train))
Accuracy.boost.train
Sensitivity.boost.train = tab.boost.train[2,2]/(tab.boost.train[2,1]+tab.boost.train[2,2])
Sensitivity.boost.train
Specificity.boost.train = tab.boost.train[1,1]/(tab.boost.train[1,1]+tab.boost.train[1,2])
Specificity.boost.train
Precision.boost.train = tab.boost.train[2,2]/(tab.boost.train[1,2]+tab.boost.train[2,2])
Precision.boost.train

auc.perf.boost=performance(prediction(as.numeric(as.character(boosting.train$pred.Car)),boosting.train$Car),"auc")
AUC.boost.train = attr(auc.perf.boost,"y.values")[[1]]
AUC.boost.train

#Validation data performance
boosting.val$pred.Car = predict(xgbModel,boosting.val.features)
tab.boost.val = table(boosting.val$Car,boosting.val$pred.Car>0.5)
tab.boost.val

Accuracy.boost.val = sum(diag(tab.boost.val)/sum(tab.boost.val))
Accuracy.boost.val
Sensitivity.boost.val = tab.boost.val[2,2]/(tab.boost.val[2,1]+tab.boost.val[2,2])
Sensitivity.boost.val
Specificity.boost.val = tab.boost.val[1,1]/(tab.boost.val[1,1]+tab.boost.val[1,2])
Specificity.boost.val
Precision.boost.val = tab.boost.val[2,2]/(tab.boost.val[1,2]+tab.boost.val[2,2])
Precision.boost.val

auc.perf.boost=performance(prediction(as.numeric(as.character(boosting.val$pred.Car)),boosting.val$Car),"auc")
AUC.boost.val = attr(auc.perf.boost,"y.values")[[1]]
AUC.boost.val

#Model Comparison
rows = c('Logit Train','Logit Test','kNN Test','Naïve Bayes Train','Naïve Bayes Test','Bagging Train','Bagging Test','Boosting Train','Boosting Test')
cols = c('Acurracy','Sensitivity','Specificity','Precision','AUC')
accuracy = c(Accuracy.logit.train,Accuracy.logit.val,Accuracy.knn,Accuracy.nb.train,Accuracy.nb.val,Accuracy.bag.train,Accuracy.bag.val,Accuracy.boost.train,Accuracy.boost.val)
sensitivity = c(Sensitivity.logit.train,Sensitivity.logit.val,Sensitivity.knn,Sensitivity.nb.train,Sensitivity.nb.val,Sensitivity.bag.train,Sensitivity.bag.val,Sensitivity.boost.train,Sensitivity.boost.val)
specificity = c(Specificity.logit.train,Specificity.logit.val,Specificity.knn,Specificity.nb.train,Specificity.nb.val,Specificity.bag.train,Specificity.bag.val,Specificity.boost.train,Specificity.boost.val)
precision = c(Precision.logit.train,Precision.logit.val,Precision.knn,Precision.nb.train,Precision.nb.val,Precision.bag.train,Precision.bag.val,Precision.boost.train,Precision.boost.val)
AUC = c(AUC.logit.train,AUC.logit.val,AUC.knn,AUC.nb.train,AUC.nb.val,AUC.bag.train,AUC.bag.val,AUC.boost.train,AUC.boost.val)

model_comparison_table = data.frame(row.names = rows,accuracy,sensitivity,specificity,precision,AUC)
model_comparison_table


