#Introduction to data analytics
#Project : Breast Cancer Detection using ID3 decision tree algorithm
#..................................................................................................................
#Setting the working directory to the location where the data set is downloaded - Breast cancer detection data set
#setwd(choose.dir())
#We can also directly mention the data set location as below commented line .If not able to chose from file system
setwd("C:\Users\ysrin\Documents\3rd year\5th sem\IDA Intro to data Analytics\Project_G12\Project_G12")

#Checking the current working directory -> Cross checking if we are in the correct working directory
getwd()
#------------------------------------------------------------------------------------------------------------------
#installing necessary packages to do the project
#..................................................................................................................
#The "plyr" package is a set of clean and consistent tools that implement the split-apply-combine pattern in R. 
install.packages("plyr")

# rpart  is used to create the decision trees in R.
install.packages("rpart")

#rpart.plot is used to plot an rpart model. A simplified interface to the prp function.
install.packages("rpart.plot")

#The tidyverse package is designed to make it easy to install and load core packages from the tidyverse in a single command.
install.packages("tidyverse")

#We use MLmetrics package for F1score
install.packages("MLmetrics")

#To perform k-fold cross validation
install.packages("caret")

#We use ggplot2 To plot ROC 
install.packages('ggplot2')
#--------------------------------------------------------------------------------------------------------------------
#Loading the installed packages to the work space
#....................................................................................................................
library(tidyverse)
library(plyr)
library(rpart)
library(rpart.plot)
library(MLmetrics)
library(caret)
library(pROC)
library(ggplot2)
#--------------------------------------------------------------------------------------------
#Loading and analyzing the data
#............................................................................................
Breast_cancer <- read.csv("BreastCancer_for_R_project.csv")
View(Breast_cancer)
str(Breast_cancer)
Breast_cancer=select(Breast_cancer, -1)
str(Breast_cancer)
#---------------------------------------------------------------------------------------------
#preprocess data 
#.............................................................................................
#Removing rows that have NA value in any attribute 
Breast_cancer <- na.omit(Breast_cancer)
str(Breast_cancer)
#Frequency of each class in the data set 
count(Breast_cancer$Class)
#Converting Class column to factor
Breast_cancer$Class <-as.factor(Breast_cancer$Class)
str(Breast_cancer)
#Storing a copy to use in implementation without library 
Breast_cancer_2<-Breast_cancer
str(Breast_cancer_2)
#--------------------------------------------------------------------------------------------
#Splitting the data into training and testing sets in 2:1 ratio
#............................................................................................
sample_split <- floor(.67*nrow(Breast_cancer))
sample_split

#setting the seed so that we get similar splitting of data when run multiple times
set.seed(1)
k =seq_len(nrow(Breast_cancer))
k
training <- sample(k,size=sample_split)
training

#Taking training data into cancer_train
cancer_train <- Breast_cancer[training, ]
cancer_train
#Taking testing data into cancer_test
cancer_test <- Breast_cancer[-training, ]
#Removing the target variable and taking remaining columns into cancer_test_data
cancer_test_data <- select(cancer_test, -10)
cancer_test_data
#Taking target variable column into cancer_test_out
cancer_test_out<-cancer_test["Class"]
cancer_test_out
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#Building a model => Using ID3 
#Using inbuilt library rpart
#............................................................................................
tree_model <- rpart(Class~.,data=cancer_train,method="class",parms=list(split='information'))
tree_model 
#------------------------------------------------------------------------------------------
#Analyzing results and plotting the tree
#..........................................................................................
printcp(tree_model)
plotcp(tree_model)
summary(tree_model)
rpart.plot(tree_model)
##
print(tree_model,cp=-1)
p <- predict(tree_model,type="class")
head(p)
View(p)
str(p)
p[1]
#CONFUSION MATRIX for training data
confusionMatrix(cancer_train$Class,p,positive='1')
#----------------------------------------------------------------------------------------
#Performance Estimation on test data
#.......................................................................................
#Checking accuracy 
predicted.classes <- tree_model %>%
  predict(cancer_test,type="class")

#Predictive accuracy
mean(predicted.classes == cancer_test$Class)
head(predicted.classes)

#Calculating F1 score
F1_Score(y_pred = predicted.classes, y_true = cancer_test$Class, positive = "1")
#Confusion Matrix 
result<-confusionMatrix(cancer_test$Class,predicted.classes,positive='1')
result
#Precision
precision <- result$byClass['Pos Pred Value']   
print(precision)
#Recall
recall <- result$byClass['Sensitivity']
print(recall)
#K fold cross validation
#Applying k-Fold Cross Validation
#TPR ->True Positive Rate | FPR -> False Positive Rate
TPR<-c()
FPR<-c()
accuracy<-c()
#Creating 10 folds
folds = createFolds(Breast_cancer$Class, k = 10)
cv = lapply(folds, function(x) {
  training_fold = Breast_cancer[-x, ]
  test_fold = Breast_cancer[x, ]
  tree_model_kfold <- rpart(Class~.,data=training_fold,method="class",parms=list(split='information'))
  test_fold_data <- select(test_fold, -10)
  #Taking target variable into test_fold_out
  test_fold_out<-test_fold["Class"]
  #---------
  y_pred <- tree_model_kfold %>%
    predict(test_fold_data,type="class")
  cm = table(test_fold[, 10], y_pred)
  acc = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  accuracy <-c(accuracy,acc)
  y= cm[1,1]/(cm[1,1]+cm[1,2])
  TPR = c(TPR,y)
  x = cm[2,1]/(cm[2,2]+cm[2,1])
  FPR = c(FPR,x)
  result<-cbind(accuracy,TPR,FPR)
  return(result)
})
#Cross validation results
b = cv
b
#Average accuracy
accuracies <-c(b$Fold01[1],b$Fold02[1],b$Fold03[1],b$Fold04[1],b$Fold05[1],b$Fold06[1],b$Fold07[1],b$Fold08[1],b$Fold09[1],b$Fold10[1])
print(accuracies)
Average_accuracy <- mean(accuracies)
print(Average_accuracy)
#Generating the ROC curve 
TPR <-c(b$Fold01[2],b$Fold02[2],b$Fold03[2],b$Fold04[2],b$Fold05[2],b$Fold06[2],b$Fold07[2],b$Fold08[2],b$Fold09[2],b$Fold10[2])
FPR <-c(b$Fold01[3],b$Fold02[3],b$Fold03[3],b$Fold04[3],b$Fold05[3],b$Fold06[3],b$Fold07[3],b$Fold08[3],b$Fold09[3],b$Fold10[3])
print(c(x=FPR,y=TPR))
#On x-axis ->FPR(False Positive Rate)
#On y-axis ->TPR(True Positive Rate)
#Plotting the ROC curve and analyzing if its good or bad
plot(FPR,TPR,
     main="ROC curve",
     type="l",
     ylab="TRP",
     xlab="FPR",
     xlim=c(0,1),
     ylim=c(0,1)
)
#Plotting the zoomed in ROC curve
plot(FPR,TPR,
     main="ROC curve",
     type="l",
     ylab="TRP",
     xlab="FPR",
)
df <-cbind.data.frame(FPR,TPR)
df
library(lubridate)
#The color bar indicate the fold number=>darkest being 1st fold and lightest being 10th fold
#Plotting the ROC points using ggplot2 with limits from 0 to 1
q<-ggplot(df, aes(FPR, TPR)) +
  geom_point(aes(col=1:10))
q+xlim(0, 1) + ylim(0, 1)
#Plotting the zoomed in ROC points plot
ggplot(df, aes(FPR, TPR)) +
  geom_point(aes(col=1:10))
#-------------------------------------------------------------------------------------------------------------#
#IMPLEMENTATION WITHOUT USING INBUILT LIBRARY
#------------------------------------------------------------------------------------------------------------#
# ID3 
#The columns are numerical in the data set
# We will implement here for continuous variables using discretization method
#We convert columns into categorical/factor variables before final implementation
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# First we will create two R class types:
# Tree and Node for easier handling of the tree-structure.
# Tree:
# Tree will be used in a recursive structure that contains  either
# another tree or a node at each branch. 
# Note: we use a list for the branches, so that each tree can contain an arbitrary number
# of brances.
tree <- function(root, branches) {
  structure(list(root=root, branches=branches), class='tree')
}
# Node:
# Node is the used for the terminal location in the trees. Each branch
# that contains a node will signify that the algorithm has either:
# 	1. Every element in the subset belongs to the same class
#	2. There are no more attributes to be selected
#	3. There are no more examples in the subset
node <- function(root) {
  structure(list(root=as.character(root)), class='node')
}

# Entropy: H(S) - a measure of uncertainty in the set S
# H(S) = - sum(p(x) * log2(p(x)) for each subset x of S
entropy <- function(S) {
  if (!is.factor(S)) S <- as.factor(S)
  
  p <- prop.table(table(S))
  
  -sum(sapply(levels(S),
              function(name) p[name] * log2(p[name]))
  )
}

# ID3: 	The main of the algorithm
#		Recursively builds a tree data structure that contains the 
#		decision tree
ID3 <- function(dataset, target_attr,
                attributes=setdiff(names(dataset), target_attr)) {
  # If there are no attributes left to classify with,
  # return the most common class left in the data set
  # as a best approximation.
  if (length(attributes) <= 0) {
    return(node(most.frq(dataset[, target_attr])))
  }
  
  # If there is only one classification left, return a
  # node with that classification as the answer
  if (length(unique(dataset[, target_attr])) == 1) {
    return(node(unique(dataset[, target_attr])[1]))
  }
  
  # Select the best attribute based on the minimum entropy
  best_attr <- attributes[which.min(sapply(attributes, entropy))]
  # Create the set of remaining attributes
  rem_attrs <- setdiff(attributes, best_attr)
  # Split the data set into groups based on levels of the best_attr
  split_dataset <- split(dataset, dataset[, best_attr])
  # Recursively branch to create the tree.
  branches <- lapply(seq_along(split_dataset), function(i) {
    # The name of the branch
    name <- names(split_dataset)[i]
    # Get branch data
    branch <- split_dataset[[i]]
    
    # If there is no data, return the most frequent class in
    # the parent, otherwise start over with new branch data.
    if (nrow(branch) == 0) node(most.frq(dataset[, target_attr]))
    else ID3(branch[, union(target_attr, rem_attrs), drop=FALSE],
             target_attr,
             rem_attrs)
  })
  names(branches) <- names(split_dataset)
  
  id3_tree <- tree(root=best_attr, branches=branches)
  id3_tree
}
most.frq <- function(nbr.class, nbr.distance) {
  uniq <- unique(nbr.class)
  uniq[which.max(tabulate(match(nbr.class, uniq)))]
}
# The prediciton method:
# This function takes a tree object created from ID3, and traverses it for
# each item in the test_obs data frame. The classifications for each item
# is returned.
predict_ID3 <- function(test_obs, id3_tree) {
  traverse <- function(obs, work_tree) {
    if (class(work_tree) == 'node') work_tree$root
    else {
      var <- work_tree$root
      new_tree <- work_tree$branches[[as.character(obs[var])]]
      traverse(obs, new_tree)
    }
  }
  apply(test_obs, 1, traverse, work_tree=id3_tree)
}
#-----------------------------------------------------------------------------------------------
#Pre-processing the Breast_cancer_2 data set - the copy that we created before
#...............................................................................................
#The model does not give good accuracy if we use all the columns that are given in the data set
#The accuracy we get is around 50% to 65% with all columns =>This is due to overfitting
#We will try to do correlation analysis and remove few columns which are highly correlated to others
str(Breast_cancer_2)
Breast_cancer_2[,1:9]
cor(Breast_cancer_2[,1:9])
#Cell shape and cell size are highly correlated hence I am removing Cell.size
str(Breast_cancer_2)
Breast_cancer_2<-select(Breast_cancer_2,-2)
str(Breast_cancer_2)
cor(Breast_cancer_2[,1:8])
#Cell shape is  moderately positively correlated with other columns -Removing Cell.shape
Breast_cancer_2<-select(Breast_cancer_2,-2)
str(Breast_cancer_2)
cor(Breast_cancer_2[,1:7])
#Bl.cromatin have correlation above 0.6 with other columns -Removing Bl.cromatin
Breast_cancer_2<-select(Breast_cancer_2,-5)
str(Breast_cancer_2)
cor(Breast_cancer_2[,1:6])
#Marg.adhesion also have correlation that is marginal with 3 columns -Removing Marg.adhesion
Breast_cancer_2<-select(Breast_cancer_2,-2)
str(Breast_cancer_2)
cor(Breast_cancer_2[,1:5])
#Converting the remaining columns into factors to apply the algorithm
Breast_cancer_2$Cl.thickness<-as.factor(Breast_cancer_2$Cl.thickness)
Breast_cancer_2$Epith.c.size<-as.factor(Breast_cancer_2$Epith.c.size)
Breast_cancer_2$Bare.nuclei<-as.factor(Breast_cancer_2$Bare.nuclei)
Breast_cancer_2$Normal.nucleoli<-as.factor(Breast_cancer_2$Normal.nucleoli)
Breast_cancer_2$Mitoses<-as.factor(Breast_cancer_2$Mitoses)
str(Breast_cancer_2)
#-----------------
#Take training data into cancer_train_2 from Breast_cancer_2
cancer_train_2 <- Breast_cancer_2[training, ]
cancer_train_2
#Taking testing data into cancer_test_2
cancer_test_2<- Breast_cancer_2[-training, ]
#Removing the target variable and taking remaining into cancer_test_data_2
cancer_test_data_2 <- select(cancer_test_2, -6)
cancer_test_data_2
#Taking target variable into cancer_test_out
cancer_test_out_2<-cancer_test_2["Class"]
cancer_test_out_2
#####Using the algorithm to build the decision tree
result3<-c()
str(cancer_train_2)
decision.tree <- ID3(cancer_train_2[-1,],"Class")
decision.tree$root
c<-decision.tree$branches
length(c)
l=length(cancer_test_data_2[,1])
l
#---------------------------------------------------------------------
#Predicting on the test data set
#----------------------------------------------------------------------
for(i in 1:l){
 z <- predict_ID3(cancer_test_data_2[i,], decision.tree)
 result3 <- c(result3,z)
}
result3
length(result3)
cancer_test_out_2[,1]
#Accuracy
mean(cancer_test_out_2==result3)
#F1_score
F1_Score(y_pred = cancer_test_out_2[,1], y_true = result3, positive ="1")
str(cancer_test_out)
str(result3)
result3 <-as.factor(result3)
str(result3)
#Confusion matrix 
result_conf<-confusionMatrix(cancer_test_out$Class,result3,positive="1")
result_conf
#Precision
precision_2 <- result_conf$byClass['Pos Pred Value']   
print(precision_2)

#Recall
recall_2 <- result_conf$byClass['Sensitivity']
print(recall_2)
#---------------------------------------------------------------
#Applying k-Fold Cross Validation
#...............................................................
#TPR_->True Positive Rate | FPR_ -> False Positive Rate
TPR_<-c()
FPR_<-c()
accuracy_<-c()
#Creating 10 folds
folds_ = createFolds(Breast_cancer_2$Class, k = 10)
cv_2 = lapply(folds, function(x) {
  training_fold = Breast_cancer_2[-x, ]
  test_fold = Breast_cancer_2[x, ]
  decision.tree <- ID3(training_fold[-1,],"Class")
  test_fold_data <- select(test_fold, -6)
  l=length(test_fold[,1])
  #Taking target variable into cancer_fold_out
  test_fold_out<-test_fold["Class"]
  result4<-c()
  for(i in 1:l){
    z <- predict_ID3(test_fold_data[i,], decision.tree)
    result4 <- c(result4,z)
  }
  cm = table(test_fold[, 6], result4)
  acc_ = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  accuracy_<-c(accuracy_,acc_)
  y_2= cm[1,1]/(cm[1,1]+cm[1,2])
  TPR_ = c(TPR_,y_2)
  x_2= cm[2,1]/(cm[2,2]+cm[2,1])
  FPR_ = c(FPR_,x_2)
  result<-cbind(accuracy_,TPR_,FPR_)
  return(result)
})
wl = cv_2
wl
#Average accuracy
accuracies_2 <-c(wl$Fold01[1],wl$Fold02[1],wl$Fold03[1],wl$Fold04[1],wl$Fold05[1],wl$Fold06[1],wl$Fold07[1],wl$Fold08[1],wl$Fold09[1],wl$Fold10[1])
print(accuracies_2)
Average_accuracy02 <- mean(accuracies_2)
print(Average_accuracy02)
#### 
TPR_ <-c(wl$Fold01[2],wl$Fold02[2],wl$Fold03[2],wl$Fold04[2],wl$Fold05[2],wl$Fold06[2],wl$Fold07[2],wl$Fold08[2],wl$Fold09[2],wl$Fold10[2])
FPR_ <-c(wl$Fold01[3],wl$Fold02[3],wl$Fold03[3],wl$Fold04[3],wl$Fold05[3],wl$Fold06[3],wl$Fold07[3],wl$Fold08[3],wl$Fold09[3],wl$Fold10[3])
print(c(x=TPR_,y=FPR_))
#Generating the ROC curve
#On x-axis ->FPR(False Positive Rate)
#On y-axis ->TPR(True Positive Rate)
#Axis limits from 0 to 1 to check how good is the model from random one
plot(FPR_,TPR_ ,
     main="ROC Plot",
     type="l",
     ylab="TRP",
     xlab="FPR",
     xlim=c(0,1),
     ylim=c(0,1)
)
#Zooming in to see the curve clearly how ROC is varying at each fold 
plot(FPR_,TPR_ ,
     main="ROC Plot",
     type="l",
     ylab="TRP",
     xlab="FPR",
)
df_2 <-cbind.data.frame(FPR_,TPR_)
df_2
library(lubridate)
#First we see how the TRP and FPR are varying at each fold
ggplot(df_2, aes(FPR_, TPR_)) +
  geom_point(aes(col=1:10))
#Comparing how good is the model ROC points from random one ->axis limits from 0 to 1
p<-ggplot(df_2, aes(FPR_, TPR_)) +
  geom_point(aes(col=1:10))
p+xlim(0, 1) + ylim(0, 1)
########################################################################################################################
                                           # END OF THE PROJECT CODE #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
