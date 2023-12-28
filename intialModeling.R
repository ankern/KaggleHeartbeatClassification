# Predictive Modeling Script
# using ecology index variables


## set working directory
setwd("~/Data Mining/Semester Project/heartbeat-sounds")
# load ecology features
mydata<-read.csv('featureExtraction_ecologyIndex.csv')


head(mydata)
mydata1<-mydata[,3:8]


# knn imputation with center and scale for pre-processing 
# knn inpute to handle missing values and centering and scaling so features are on the same scale
library(caret)
library(RANN)
preProcValues <- preProcess(mydata1, method = c("knnImpute"))
d_imp <- predict(preProcValues, mydata1)

# write imputed data 
write.csv(d_imp,file='imputedFeatures.csv')

# visualize features by class
par(mfrow=c(3,2))
for (i in 2:6){
  boxplot(d_imp[,i]~d_imp[,1],main=colnames(d_imp)[i])
}

# remove unlabled samples for model training
yTmp<-d_imp[d_imp$class!='Aunl',1]
xTmp<-d_imp[d_imp$class!='Aunl',2:6]

yTmp<-factor(yTmp)

#### data splitting (stratified random sampling)
set.seed(221)
train<-createDataPartition(yTmp,times=1,p=.75, list=FALSE)

# subset data
xTrain<-xTmp[train,]
yTrain<-yTmp[train]
xTest<-xTmp[-train,]
yTest<-yTmp[-train]


# show distributions of training and testing data
par(mfrow=c(2,1))
barplot(table(yTrain),ylim=c(0,35),main='Training Data Classes')
barplot(table(yTest),ylim=c(0,35),main = 'Test Data Classes')

# set train control, 10 fold CV repeated 10 times
set.seed(221)
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 10,
     classProbs = TRUE,
      savePredictions = TRUE)


## pre-allocate space for accuracy metrics output
output<-matrix(nrow=8,ncol=10)
colnames(output)<-c('Model','Overall Accuracy','Arti Sens','Arti Spec','Extr Sens',
                    'Extr Spec','Murm Sens','Murm Spec','Norm Sens','Norm Spec')

output[,1]<-c('Partial Least Squares','GLMN','Linear Discriminant Analysis','Neural Network',
              'Support Vector Machine','Classification Tree',
              'Random Forest','Naive Bayes')

# pre-allocate space for precision of each model by class
precisionTable<-matrix(nrow=8,ncol=4)
colnames(precisionTable)<-c('Artifact','Extra Sound','Murmur','Normal')

######### Partial Least Squares Discriminant Analysis ################
# linear method 
set.seed(221)

plsFit<-train(xTrain,yTrain,
              method = 'pls',
              tuneGrid = expand.grid(.ncomp = 1:5), # tuning parameters
              metric = "Accuracy",
              trControl=ctrl)       # for the cross validation, set and described above


# predict on training data (not for report but just for an internal check)
pred<-predict(plsFit,xTrain)
confusionMatrix(pred,yTrain)


# predict and calculate accuracy metrics when predicting on test set
pred<-predict(plsFit,xTest)
CM<-confusionMatrix(pred,yTest)

## tabulate accuracy metrics
output[1,2]<-CM$overall[1]

output[1,3]<-CM$byClass[1,1]; output[1,4]<-CM$byClass[1,2]; output[1,5]<-CM$byClass[2,1]
output[1,6]<-CM$byClass[2,2]; output[1,7]<-CM$byClass[3,1]; output[1,8]<-CM$byClass[3,2]
output[1,9]<-CM$byClass[4,1]; output[1,10]<-CM$byClass[4,2]

precisionTable[1,1]<-CM$table[1,1]/sum(CM$table[1,])
precisionTable[1,2]<-CM$table[2,2]/sum(CM$table[2,])
precisionTable[1,3]<-CM$table[3,3]/sum(CM$table[3,])
precisionTable[1,4]<-CM$table[4,4]/sum(CM$table[4,])


###### Penalized Models ##########
# Lasso and Elastic-Net Regularized Generalized Linear Models
set.seed(221)

# set grid of tuning parameters
glmnGrid<-expand.grid(.alpha = c(0,.1,.2,.4,.6,.8,1),
                      .lambda = seq(.01,.2, length=5))

glmnTuned<-train(xTrain,yTrain,
                 method = 'glmnet',
                 tuneGrid = glmnGrid,
                 metric = 'Accuracy',
                 trControl = ctrl)

glmnTuned
pred<-predict(glmnTuned,xTrain)
confusionMatrix(pred,yTrain)

#summary(glmnTuned)
pred<-predict(glmnTuned,xTest)
CM<-confusionMatrix(pred,yTest)

#Accuracy
output[2,2]<-CM$overall[1]

output[2,3]<-CM$byClass[1,1]; output[2,4]<-CM$byClass[1,2]; output[2,5]<-CM$byClass[2,1]
output[2,6]<-CM$byClass[2,2]; output[2,7]<-CM$byClass[3,1]; output[2,8]<-CM$byClass[3,2]
output[2,9]<-CM$byClass[4,1]; output[2,10]<-CM$byClass[4,2]

precisionTable[2,1]<-CM$table[1,1]/sum(CM$table[1,])
precisionTable[2,2]<-CM$table[2,2]/sum(CM$table[2,])
precisionTable[2,3]<-CM$table[3,3]/sum(CM$table[3,])
precisionTable[2,4]<-CM$table[4,4]/sum(CM$table[4,])



######### Linear Discriminant Analysis ##################
set.seed(221)
ldaFit1<-train(xTrain,yTrain,
               method='lda',
               metric='Accuracy',
               trControl=ctrl)

pred<-predict(ldaFit1,xTrain)
confusionMatrix(pred,yTrain)

summary(ldaFit1)
pred<-predict(ldaFit1,xTest)
CM<-confusionMatrix(pred,yTest)



#Accuracy
output[3,2]<-CM$overall[1]

output[3,3]<-CM$byClass[1,1]; output[3,4]<-CM$byClass[1,2]; output[3,5]<-CM$byClass[2,1]
output[3,6]<-CM$byClass[2,2]; output[3,7]<-CM$byClass[3,1]; output[3,8]<-CM$byClass[3,2]
output[3,9]<-CM$byClass[4,1]; output[3,10]<-CM$byClass[4,2]


precisionTable[3,1]<-CM$table[1,1]/sum(CM$table[1,])
precisionTable[3,2]<-CM$table[2,2]/sum(CM$table[2,])
precisionTable[3,3]<-CM$table[3,3]/sum(CM$table[3,])
precisionTable[3,4]<-CM$table[4,4]/sum(CM$table[4,])



############ Neural Network #######
set.seed(221)

# set grid for tuning parameters, we can cite a reference for this 
# http://www.springer.com/us/book/9781461468486 
nnetGrid <- expand.grid(.size = 1:10,
                        .decay = c(0,.1,1,2))
maxSize <- max(nnetGrid$.size)
numWts <- 1*(maxSize*(length(xTrain)+1)+maxSize+1)

nnetFit<-train(xTrain,yTrain,
               method = "nnet",
               metric = "Accuracy",
               tuneGrid = nnetGrid,
               trace = FALSE,
               maxit = 50,
               MaxNWts =numWts,
               trControl = ctrl)

plot(nnetFit, main ='Tuning Parameter for Neural Network')

nnetFit
pred<-predict(nnetFit,xTrain)
confusionMatrix(pred,yTrain)

summary(nnetFit)
pred<-predict(nnetFit,xTest)
CM<-confusionMatrix(pred,yTest)

#Accuracy
output[4,2]<-CM$overall[1]

output[4,3]<-CM$byClass[1,1]; output[4,4]<-CM$byClass[1,2]; output[4,5]<-CM$byClass[2,1]
output[4,6]<-CM$byClass[2,2]; output[4,7]<-CM$byClass[3,1]; output[4,8]<-CM$byClass[3,2]
output[4,9]<-CM$byClass[4,1]; output[4,10]<-CM$byClass[4,2]

precisionTable[4,1]<-CM$table[1,1]/sum(CM$table[1,])
precisionTable[4,2]<-CM$table[2,2]/sum(CM$table[2,])
precisionTable[4,3]<-CM$table[3,3]/sum(CM$table[3,])
precisionTable[4,4]<-CM$table[4,4]/sum(CM$table[4,])

# look at relative variable importance, this varImp function will work for all models
# in this script 
varImp(nnetFit)

############# Support Vector Machine ##################
set.seed(221)

library(kernlab)
# set tuning paramteters, use same reference book as above
sigmaRange<- sigest(as.matrix(xTrain))
svmRGrid <- expand.grid(.sigma = sigmaRange[1],
                        .C = 2^(seq(-4, 16)))
svmRModel <- train(x = xTrain, 
                   y = yTrain,
                   method = "svmRadial",
                   metric = "Accuracy",
                   tuneGrid = svmRGrid,
                   fit = FALSE,
                   trControl = ctrl)
svmRModel

plot(svmRModel, main='Tuning Cost Parameter for SVM')

pred<-predict(svmRModel,xTrain)
confusionMatrix(pred,yTrain)

pred<-predict(svmRModel,xTest)
CM<-confusionMatrix(pred,yTest)

#Accuracy
output[5,2]<-CM$overall[1]

output[5,3]<-CM$byClass[1,1]; output[5,4]<-CM$byClass[1,2]; output[5,5]<-CM$byClass[2,1]
output[5,6]<-CM$byClass[2,2]; output[5,7]<-CM$byClass[3,1]; output[5,8]<-CM$byClass[3,2]
output[5,9]<-CM$byClass[4,1]; output[5,10]<-CM$byClass[4,2]

precisionTable[5,1]<-CM$table[1,1]/sum(CM$table[1,])
precisionTable[5,2]<-CM$table[2,2]/sum(CM$table[2,])
precisionTable[5,3]<-CM$table[3,3]/sum(CM$table[3,])
precisionTable[5,4]<-CM$table[4,4]/sum(CM$table[4,])



########## Classification Trees ##############
set.seed(221)
rpartFit<-train(xTrain,yTrain,
                method = "rpart",
                tuneLength = 5,
                metric = "Accuracy",
                trControl = ctrl)
rpartFit
plot(rpartFit,main = 'Complexity of Tree for Pruning Application')

# rpartFit automatically selected the best tree, no need to manually prune
plot(rpartFit,main="Tuning Paramter for 
     Classification Tree",ylab = "Cross-Validated Accuracy")

pred<-predict(rpartFit,xTrain)
confusionMatrix(pred,yTrain)

require(rpart.plot)
rpart.plot(rpartFit$finalModel,type=0,extra=1,cex=NULL,varlen=20)

pred<-predict(rpartFit,xTest)
CM<-confusionMatrix(pred,yTest)

#Accuracy
output[6,2]<-CM$overall[1]

output[6,3]<-CM$byClass[1,1]; output[6,4]<-CM$byClass[1,2]; output[6,5]<-CM$byClass[2,1]
output[6,6]<-CM$byClass[2,2]; output[6,7]<-CM$byClass[3,1]; output[6,8]<-CM$byClass[3,2]
output[6,9]<-CM$byClass[4,1]; output[6,10]<-CM$byClass[4,2]

precisionTable[6,1]<-CM$table[1,1]/sum(CM$table[1,])
precisionTable[6,2]<-CM$table[2,2]/sum(CM$table[2,])
precisionTable[6,3]<-CM$table[3,3]/sum(CM$table[3,])
precisionTable[6,4]<-CM$table[4,4]/sum(CM$table[4,])



############## Random Forest ################

library(caret)
set.seed(221)
rfFit<-train(xTrain,yTrain,
             method = "rf",
             trControl =ctrl,
             metric = "Accuracy",
             importance = TRUE,
             verbose = TRUE,
             tuneGrid = data.frame(mtry=c(1:20))) # tuning parameter (# of trees to make)

rfFit
plot(rfFit) # selected 1 tree which indicates random forest will not perform better than
            # the classification tree, probably due to the small number of features we started with
pred<-predict(rfFit,xTrain)
confusionMatrix(pred,yTrain)

pred<-predict(rfFit,xTest)
CM<-confusionMatrix(pred,yTest)

#Accuracy
output[7,2]<-CM$overall[1]

output[7,3]<-CM$byClass[1,1]; output[7,4]<-CM$byClass[1,2]; output[7,5]<-CM$byClass[2,1]
output[7,6]<-CM$byClass[2,2]; output[7,7]<-CM$byClass[3,1]; output[7,8]<-CM$byClass[3,2]
output[7,9]<-CM$byClass[4,1]; output[7,10]<-CM$byClass[4,2]

precisionTable[7,1]<-CM$table[1,1]/sum(CM$table[1,])
precisionTable[7,2]<-CM$table[2,2]/sum(CM$table[2,])
precisionTable[7,3]<-CM$table[3,3]/sum(CM$table[3,])
precisionTable[7,4]<-CM$table[4,4]/sum(CM$table[4,])



## Naive Bayes

set.seed(221)
nbFit<-train(xTrain,yTrain,
             method="nb",
             metric="Accuracy",
             trControl=ctrl)

nbFit
pred<-predict(nbFit,xTrain)
confusionMatrix(pred,yTrain)

pred<-predict(nbFit,xTest)
CM<-confusionMatrix(pred,yTest)



#Accuracy
output[8,2]<-CM$overall[1]

output[8,3]<-CM$byClass[1,1]; output[8,4]<-CM$byClass[1,2]; output[8,5]<-CM$byClass[2,1]
output[8,6]<-CM$byClass[2,2]; output[8,7]<-CM$byClass[3,1]; output[8,8]<-CM$byClass[3,2]
output[8,9]<-CM$byClass[4,1]; output[8,10]<-CM$byClass[4,2]

precisionTable[8,1]<-CM$table[1,1]/sum(CM$table[1,])
precisionTable[8,2]<-CM$table[2,2]/sum(CM$table[2,])
precisionTable[8,3]<-CM$table[3,3]/sum(CM$table[3,])
precisionTable[8,4]<-CM$table[4,4]/sum(CM$table[4,])

### Write out results
setwd("~/Data Mining/Semester Project")
write.csv(output,file='modelOutput.csv')

write.csv(precisionTable,file='modelPrecision.csv')

# make predictions on unlabled data with top four models
predLDA<-predict(ldaFit1,d_imp[d_imp$class=='Aunl',2:6])
predNNET<-predict(nnetFit,d_imp[d_imp$class=='Aunl',2:6])
predSVM<-predict(svmRModel,d_imp[d_imp$class=='Aunl',2:6])
predCTree<-predict(rpartFit,d_imp[d_imp$class=='Aunl',2:6])


AunlPred<-cbind(table(predLDA),table(predNNET),table(predSVM),table(predCTree))
colnames(AunlPred)<-c('LDA','NNET','SVM','Tree')
write.csv(AunlPred,file='AunlPred.csv')



