rm(list=ls())
setwd("~/FYP/model/xgboost/xgboost complete cases/Boruta")
library(mlbench)
library(caret)
library(doParallel)
library(tidyverse)
library(xgboost)
library(pROC)
library(SHAPforxgboost)
library(Matrix)

set.seed(1234)

########## #############################boruta #######################################
library(Boruta)
xgb.boruta=Boruta(training.df[,-55],
                  training.df$status,
                  maxRuns=100, 
                  doTrace=2,
                  holdHistory=TRUE,
                  getImp=getImpXgboost,
                  max.depth=xgb.train$bestTune$max_depth, 
                  eta=xgb.train$bestTune$eta, 
                  nthread=4, 
                  min_child_weight=xgb.train$bestTune$min_child_weight,
                  scale_pos_weight=sumwneg/sumwpos, 
                  eval_metric="auc", 
                  gamma=xgb.train$bestTune$gamma,
                  nrounds=xgb.crv$best_iteration, 
                  objective="binary:logistic",
                  tree_method="hist",
                  lambda=0,
                  alpha=0)

boruta_dec=attStats(xgb.boruta)
boruta_dec[boruta_dec$decision!="Rejected",]
boruta_dec

# ptageatnotification 0.07697227 0.07845304 0.06213897 0.09291840        1 Confirmed
# heartrate           0.06197601 0.06688507 0.03694404 0.08220039        1 Confirmed
# bpsys               0.06224979 0.06366182 0.05248782 0.07112407        1 Confirmed
# killipclass         0.24363017 0.24743403 0.22498313 0.25822167        1 Confirmed
# ck                  0.10946616 0.11228357 0.08840906 0.12784845        1 Confirmed
# fbg                 0.10013067 0.10099960 0.08693823 0.11269160        1 Confirmed
# bb                  0.05453467 0.05554839 0.04771580 0.05874016        1 Confirmed
# acei                0.05468362 0.05590332 0.04677812 0.05960163        1 Confirmed
# oralhypogly         0.04531158 0.04626761 0.03889840 0.04978886        1 Confirmed


############## run selected variables from boruta #############
library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")


train.ind=createDataPartition(ds$status, times = 1, p=0.7, list = FALSE)
training.df=ds[train.ind, ]
testing.df=ds[-train.ind, ]
# write.csv(testing.df,'testing.dfborutacompletecases.csv')

training.df<- subset(training.df, select = c(ptageatnotification,heartrate,bpsys,
                                             killipclass, ck,fbg, bb, acei,
                                             oralhypogly,status))
testing.df<- subset(testing.df, select = c(ptageatnotification,heartrate,bpsys,
                                           killipclass, ck,fbg, bb, acei,
                                           oralhypogly,status))

training.df$status <- ifelse(test = training.df$status ==1, yes = "1", no = "0")
testing.df$status <- ifelse(test = testing.df$status ==1, yes = "1", no = "0")

training.df$status <- as.numeric(training.df$status)
testing.df$status <- as.numeric(testing.df$status)

label=as.numeric(as.character(training.df$status))
ts_label=as.numeric(as.character(testing.df$status))

sumwpos=sum(label==1)
sumwneg=sum(label==0)
print(sumwneg/sumwpos)

dtest=sparse.model.matrix(status~.-1, data = data.frame(testing.df))
dtrain=sparse.model.matrix(status~.-1, training.df)

xgb.grid=expand.grid(nrounds=c(25,50,100),
                     eta=c(0.01,0.05),
                     max_depth=3,
                     gamma = 0,
                     subsample = c(0.7,0.8,0.9),
                     min_child_weight = c(1,2),
                     colsample_bytree = 1)

myCl=makeCluster(detectCores()-1)
registerDoParallel(myCl)

xgb.control=trainControl(method = "cv",
                         number = 5,
                         verboseIter = TRUE,
                         returnData = FALSE,
                         returnResamp = "none",
                         classProbs = TRUE,
                         allowParallel = TRUE)

xgb.train = train(x = training.df,
                  y = label,
                  trControl = xgb.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  scale_pos_weight=sumwneg/sumwpos)

stopCluster(myCl)

xgb.train$bestTune
# nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
# 1     500         3 0.01     0                1                1       0.7

params=list("eta"=xgb.train$bestTune$eta,
            "max_depth"=xgb.train$bestTune$max_depth,
            "gamma"=xgb.train$bestTune$gamma,
            "min_child_weight"=xgb.train$bestTune$min_child_weight,
            "nthread"=4,
            "objective"="binary:logistic")

xgb.crv=xgb.cv(params = params,
               data = dtrain,
               nrounds = 5000,
               nfold = 10,
               label = label,
               showsd = TRUE,
               metrics = "auc",
               stratified = TRUE,
               verbose = TRUE,
               print_every_n = 1L,
               early_stopping_rounds = 100,
               scale_pos_weight=sumwneg/sumwpos)

xgb.crv$best_iteration

xgb.boruta=xgboost(data = dtrain,
                     label = label,
                     max.depth=xgb.train$bestTune$max_depth,
                     eta=xgb.train$bestTune$eta,
                     nthread=4,
                     min_child_weight=xgb.train$bestTune$min_child_weight,
                     scale_pos_weight=sumwneg/sumwpos,
                     eval_metric="auc",
                     nrounds=xgb.crv$best_iteration,
                     objective="binary:logistic")

importance=xgb.importance(feature_names = colnames(dtrain), model = xgb.boruta)
importance$Feature[1:9]
(gg=xgb.ggplot.importance(importance_matrix = importance[1:9,]))


result.predicted1 <- predict(xgb.boruta, newdata = dtest, type = "prob")

result.roc1 <- roc(ts_label, result.predicted1)
result.roc1
# Area under the curve: 0.85

result.predicted1 <- ifelse (result.predicted1 > 0.5,1,0)
result.predicted1 <- as.factor(result.predicted1)

cm_all_1 <- confusionMatrix (result.predicted1,
                             as.factor(ts_label))
cm_all_1

# 
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1
# 0   69  186
# 1   30 1006
# 
# Accuracy : 0.8327          
# 95% CI : (0.8112, 0.8527)
# No Information Rate : 0.9233          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.314           
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.69697         
#             Specificity : 0.84396         
#          Pos Pred Value : 0.27059         
#          Neg Pred Value : 0.97104         
#              Prevalence : 0.07668         
#          Detection Rate : 0.05345         
#    Detection Prevalence : 0.19752         
#       Balanced Accuracy : 0.77046         
#                                           
#        'Positive' Class : 0    

saveRDS(xgb.boruta,"xgbboruta.rds")
modelboruta = readRDS("xgbboruta.rds")

#####################################################################################################################
##################### testing stemi boruta using complete cases testing set ######################
stemi <- read_csv("testing.dfboruta - stemi.csv")

#stemi<- c(ptageatnotification,heartrate,bpsys,killipclass, ck,fbg, bb, acei,oralhypogly,status)

stemi<- subset(stemi, select = c(ptageatnotification,heartrate,bpsys,
                                 killipclass, ck,fbg, bb, acei,
                                 oralhypogly,status))

stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

############### different method to plot roc #####################
stemipred <- predict (modelboruta,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
# Area under the curve: 0.8496

#predict
xgbooststemi = xgb.DMatrix(data=xstemi, label=ystemi)
predictionstemi = predict(modelboruta, newdata = xgbooststemi )
predictionstemi = ifelse(predictionstemi > 0.5, 1, 0)
predictionstemi
table(predictionstemi)

####### confusion matrix ##########
predictionstemi = as.factor(predictionstemi) 
cmstemi = confusionMatrix(predictionstemi, as.factor(ystemi), positive = '0')
cmstemi
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0  47 107
# 1  13 396
# 
# Accuracy : 0.7869        
# 95% CI : (0.7507, 0.82)
# No Information Rate : 0.8934        
# P-Value [Acc > NIR] : 1             
# 
# Kappa : 0.3377        
# 
# Mcnemar's Test P-Value : <2e-16        
#                                         
#             Sensitivity : 0.78333       
#             Specificity : 0.78728       
#          Pos Pred Value : 0.30519       
#          Neg Pred Value : 0.96822       
#              Prevalence : 0.10657       
#          Detection Rate : 0.08348       
#    Detection Prevalence : 0.27353       
#       Balanced Accuracy : 0.78530       
#                                         
#        'Positive' Class : 0             
#                              

##################### testing stemi boruta using complete cases testing set ################
nstemi <- read_csv("testing.dfboruta - nstemi.csv")


nstemi<- subset(nstemi, select = c(ptageatnotification,heartrate,bpsys,
                                   killipclass, ck,fbg, bb, acei,
                                   oralhypogly,status))

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) #remove status

############### different method to plot roc #####################
nstemipred <- predict (modelboruta,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
# Area under the curve: 0.8356

#predict
xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)
predictionnstemi = predict(modelboruta, newdata = xgboostnstemi )
predictionnstemi = ifelse(predictionnstemi > 0.5, 1, 0)
predictionnstemi
table(predictionnstemi)

####### confusion matrix ##########
predictionnstemi = as.factor(predictionnstemi) 
cmnstemi = confusionMatrix(predictionnstemi, as.factor(ynstemi), positive = '0')
cmnstemi
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0  22  79
# 1  17 610
# 
# Accuracy : 0.8681          
# 95% CI : (0.8414, 0.8919)
# No Information Rate : 0.9464          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.2568          
# 
# Mcnemar's Test P-Value : 4.791e-10       
#                                           
#             Sensitivity : 0.56410         
#             Specificity : 0.88534         
#          Pos Pred Value : 0.21782         
#          Neg Pred Value : 0.97289         
#              Prevalence : 0.05357         
#          Detection Rate : 0.03022         
#    Detection Prevalence : 0.13874         
#       Balanced Accuracy : 0.72472         
#                                           
#        'Positive' Class : 0 

#########################################################################################################################
########################## testing stemi using dataset not from complete cases
stemi <- read_csv("testdatasetelderly - stemi.csv")
modelboruta = readRDS("xgbboruta.rds")

#stemi<- c(ptageatnotification,heartrate,bpsys,killipclass, ck,fbg, bb, acei,oralhypogly,status)

stemi<- subset(stemi, select = c(ptageatnotification,heartrate,bpsys,
                                 killipclass, ck,fbg, bb, acei,
                                 oralhypogly,status))

stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

############### different method to plot roc #####################
stemipred <- predict (modelboruta,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
# Area under the curve: 0.8825

ci.auc(as.vector(ystemi), as.vector(stemipred))
# 95% CI: 0.844-0.921 (DeLong)

#predict
xgbooststemi = xgb.DMatrix(data=xstemi, label=ystemi)
predictionstemi = predict(modelboruta, newdata = xgbooststemi )
predictionstemi = ifelse(predictionstemi > 0.5, 1, 0)
predictionstemi
table(predictionstemi)

####### confusion matrix ##########
predictionstemi = as.factor(predictionstemi) 
cmstemi = confusionMatrix(predictionstemi, as.factor(ystemi), positive = '0')
cmstemi
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0  58 111
# 1   9 414
# 
# Accuracy : 0.7973         
# 95% CI : (0.7626, 0.829)
# No Information Rate : 0.8868         
# P-Value [Acc > NIR] : 1              
# 
# Kappa : 0.3932         
# 
# Mcnemar's Test P-Value : <2e-16         
#                                          
#             Sensitivity : 0.86567        
#             Specificity : 0.78857        
#          Pos Pred Value : 0.34320        
#          Neg Pred Value : 0.97872        
#              Prevalence : 0.11318        
#          Detection Rate : 0.09797        
#    Detection Prevalence : 0.28547        
#       Balanced Accuracy : 0.82712        
#                                          
#        'Positive' Class : 0  

########################## testing nstemi using dataset not from complete cases##########
nstemi <- read_csv("testdatasetelderly - nstemi.csv")

nstemi<- subset(nstemi, select = c(ptageatnotification,heartrate,bpsys,
                                   killipclass, ck,fbg, bb, acei,
                                   oralhypogly,status))

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) #remove status

############### different method to plot roc #####################
nstemipred <- predict (modelboruta,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
# Area under the curve: 0.8989

ci.auc(as.vector(ynstemi), as.vector(nstemipred))
# 95% CI: 0.8486-0.9492 (DeLong)

#predict
xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)
predictionnstemi = predict(modelboruta, newdata = xgboostnstemi )
predictionnstemi = ifelse(predictionnstemi > 0.5, 1, 0)
predictionnstemi
table(predictionnstemi)

####### confusion matrix ##########
predictionnstemi = as.factor(predictionnstemi) 
cmnstemi = confusionMatrix(predictionnstemi, as.factor(ynstemi), positive = '0')
cmnstemi
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0  28  75
# 1  11 585
# 
# Accuracy : 0.877           
# 95% CI : (0.8503, 0.9004)
# No Information Rate : 0.9442          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.341           
# 
# Mcnemar's Test P-Value : 1.095e-11       
#                                           
#             Sensitivity : 0.71795         
#             Specificity : 0.88636         
#          Pos Pred Value : 0.27184         
#          Neg Pred Value : 0.98154         
#              Prevalence : 0.05579         
#          Detection Rate : 0.04006         
#    Detection Prevalence : 0.14735         
#       Balanced Accuracy : 0.80216         
#                                           
#        'Positive' Class : 0   