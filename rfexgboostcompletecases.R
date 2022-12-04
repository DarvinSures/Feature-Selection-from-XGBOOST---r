#Recursive Feature Elimination (RFE)
rm(list=ls())
setwd("~/FYP/model/xgboost/xgboost complete cases/rfe")
library(mlbench)
library(caret)
library(doParallel)
library(tidyverse)
library(xgboost)
library(pROC)
library(SHAPforxgboost)
library(Matrix)
library(Boruta)

##
set.seed(1234)

#import dataset
library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")
ds <- ds[,-1] #remove patient id
ds<- subset(ds, select = -c(timiscorestemi,timiscorenstemi,acsstratum))

#second time
# ds <- subset(ds,select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",
#                            "ldlc","tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
#                            "hdlc","calcantagonist","arb","antiarr","statin","ecgabnormtypestelev2",
#                            "cardiaccath","ecgabnormtypestelev1","ecgabnormlocational","ecgabnormtypestdep","status"))

# third time
# ds <- subset(ds,select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",
#                             "tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
#                            "ldlc","hdlc","antiarr","statin","ecgabnormtypestelev2",
#                             "cardiaccath","status"))


#spliting
train.ind=createDataPartition(ds$status, times = 1, p=0.7, list = FALSE)
training.df=ds[train.ind, ]
testing.df=ds[-train.ind, ]
# write.csv(testing.df, "rfetesting.df.csv")

label=as.numeric(as.character(training.df$status))
ts_label=as.numeric(as.character(testing.df$status))

dtest=sparse.model.matrix(status~.-1, data = data.frame(testing.df))
dtrain=sparse.model.matrix(status~.-1, training.df)

#all var
# categorical_var <- c("ptsex","ptrace","smokingstatus","cdys","cdm","chpt","cpremcvd",
#                      "cmi","ccap","canginamt2wk","canginapast2wk","cheartfail","clung",
#                      "crenal","ccerebrovascular","cpvascular","killipclass","ecgabnormtypestelev1",
#                      "ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
#                      "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational",
#                      "ecgabnormlocationll","ecgabnormlocationtp","ecgabnormlocationrv",
#                      "cardiaccath","pci","cabg","asa","gpri","heparin","lmwh",
#                      "bb","acei","arb","statin","lipidla","diuretic","calcantagonist",
#                      "oralhypogly","insulin","antiarr","status")

#second one
# categorical_var <- c("killipclass","oralhypogly","bb", "acei","crenal",
#                      "cdm","insulin","calcantagonist","arb",
#                      "antiarr","statin","ecgabnormtypestelev2",
#                      "cardiaccath","ecgabnormtypestelev1","ecgabnormlocational","ecgabnormtypestdep","status")

# #third one
# categorical_var <- c("killipclass","oralhypogly","bb","acei","crenal",
#                      "cdm","insulin","antiarr","statin","ecgabnormtypestelev2",
#                      "cardiaccath","status")



training.df[,categorical_var] <- lapply(training.df[,categorical_var],factor)
testing.df[,categorical_var] <- lapply(testing.df[,categorical_var],factor)

set.seed(1234)
options(warn=-1)

subsets <- c(1:54)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 3,
                   verbose = FALSE)
myCl=makeCluster(detectCores()-1)
registerDoParallel(myCl)


lmProfile <- rfe(x=training.df[,-21], y=training.df$status,
                 sizes = subsets,
                 rfeControl = ctrl)
stopCluster(myCl)

lmProfile$optVariables
# all
# [1] "killipclass"          "fbg"                  "bpsys"                "oralhypogly"
# [5] "bb"                   "heartrate"            "ck"                   "ldlc"
# [9] "tc"                   "acei"                 "crenal"               "bpdias"
# [13] "ptageatnotification"  "cdm"                  "insulin"              "hdlc"
# [17] "calcantagonist"       "arb"                  "antiarr"              "statin"
# [21] "ecgabnormtypestelev2" "cardiaccath"          "ecgabnormtypestelev1" "ecgabnormlocational"
# [25] "ecgabnormtypestdep"

# 2nd
# [1] "killipclass"          "fbg"                  "bpsys"                "oralhypogly"          "bb"
# [6] "ldlc"                 "ck"                   "heartrate"            "tc"                   "acei"
# [11] "bpdias"               "crenal"               "cdm"                  "ptageatnotification"  "statin"
# [16] "insulin"              "hdlc"                 "antiarr"              "ecgabnormtypestelev2" "cardiaccath"

#3rd
# [1] "killipclass"          "fbg"                  "oralhypogly"          "bpsys"                "bb"
# [6] "ldlc"                 "tc"                   "ck"                   "heartrate"            "acei"
# [11] "bpdias"               "crenal"               "cdm"                  "ptageatnotification"  "statin"
# [16] "cardiaccath"          "insulin"              "antiarr"              "ecgabnormtypestelev2" "hdlc"

lmProfile$metric
# "Accuracy"

trellis.par.set(caretTheme())
plot1 <- plot(lmProfile, type = c("g", "o"))
plot2 <- plot(lmProfile, type = c("g", "o"), metric = "Kappa")
print(plot1, split=c(1,1,1,2), more=TRUE)
print(plot2, split=c(1,2,1,2))


########################################################################################################################
#xgboost run with variable selected from RFE
#SELECTED FROM ALL VAR
library(mlbench)
library(caret)
library(doParallel)
library(tidyverse)
library(xgboost)
library(pROC)
library(SHAPforxgboost)
library(Matrix)

set.seed(1234)

library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")
ds <- ds[,-1]
ds<- subset(ds, select = -c(timiscorestemi,timiscorenstemi,acsstratum))
ds <- subset(ds, select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",    
                            "ldlc","tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
                             "hdlc","calcantagonist","arb","antiarr","statin","ecgabnormtypestelev2",
                             "cardiaccath","ecgabnormtypestelev1","ecgabnormlocational","ecgabnormtypestdep","status"))

ds$status <- ifelse(test = ds$status ==1, yes = "1", no = "0")
ds$status <- as.numeric(ds$status)

train.ind=createDataPartition(ds$status, times = 1, p=0.7, list = FALSE)
training.df=ds[train.ind, ]

#save testing.df into another excel file
testing.df=ds[-train.ind, ]
#write.csv(testing.df,'testing.dfcompletecases.csv')

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
                     min_child_weight = c(1, 2), 
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
                  y = factor(label, labels = c("Dead", "Alive")),
                  trControl = xgb.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  scale_pos_weight=sumwneg/sumwpos)


stopCluster(myCl)

params=list("eta"=xgb.train$bestTune$eta,
            "max_depth"=xgb.train$bestTune$max_depth,
            "gamma"=xgb.train$bestTune$gamma,
            "min_child_weight"=xgb.train$bestTune$min_child_weight,
            "nthread"=4,
            "objective"="binary:logistic")

label <- as.numeric(label)

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

xgb.rfe1=xgboost(data = dtrain, 
                label = label, 
                max.depth=xgb.train$bestTune$max_depth, 
                eta=xgb.train$bestTune$eta, 
                nthread=4, 
                min_child_weight=xgb.train$bestTune$min_child_weight,
                scale_pos_weight=sumwneg/sumwpos, 
                eval_metric="auc", 
                nrounds=xgb.crv$best_iteration, 
                objective="binary:logistic")

# saveRDS(xgb.rfe1,"xgbrfe1.rds")
# model = readRDS("xgbrfe1.rds")

result.predicted1 <- predict(xgb.rfe1, newdata = dtest, type = "prob")

result.roc1 <- roc(ts_label, result.predicted1)
result.roc1
# Area under the curve: 0.849

result.predicted1 <- ifelse (result.predicted1 > 0.5,1,0)
result.predicted1 <- as.factor(result.predicted1)

cm_all_1 <- confusionMatrix (result.predicted1,
                             as.factor(ts_label))
cm_all_1
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1
# 0   66  180
# 1   33 1012
# 
# Accuracy : 0.835           
# 95% CI : (0.8136, 0.8549)
# No Information Rate : 0.9233          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3068          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.66667         
#             Specificity : 0.84899         
#          Pos Pred Value : 0.26829         
#          Neg Pred Value : 0.96842         
#              Prevalence : 0.07668         
#          Detection Rate : 0.05112         
#    Detection Prevalence : 0.19055         
#       Balanced Accuracy : 0.75783         
#                                           
#        'Positive' Class : 0   


########################################################################################################################
#xgboost run with variable selected from RFE
#SELECTED FROM SECOND CYCLE

set.seed(1234)

library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")
ds <- ds[,-1]
ds<- subset(ds, select = -c(timiscorestemi,timiscorenstemi,acsstratum))
ds <- subset(ds, select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",
                            "tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
                            "ldlc","hdlc","antiarr","statin","ecgabnormtypestelev2",
                            "cardiaccath","status"))

ds$status <- ifelse(test = ds$status ==1, yes = "1", no = "0")
ds$status <- as.numeric(ds$status)

train.ind=createDataPartition(ds$status, times = 1, p=0.7, list = FALSE)
training.df=ds[train.ind, ]

#save testing.df into another excel file
testing.df=ds[-train.ind, ]
#write.csv(testing.df,'testing.dfcompletecases.csv')

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
                     min_child_weight = c(1, 2), 
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
                  y = factor(label, labels = c("Dead", "Alive")),
                  trControl = xgb.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  scale_pos_weight=sumwneg/sumwpos)


stopCluster(myCl)

params=list("eta"=xgb.train$bestTune$eta,
            "max_depth"=xgb.train$bestTune$max_depth,
            "gamma"=xgb.train$bestTune$gamma,
            "min_child_weight"=xgb.train$bestTune$min_child_weight,
            "nthread"=4,
            "objective"="binary:logistic")

label <- as.numeric(label)

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

xgb.rfe2=xgboost(data = dtrain, 
                label = label, 
                max.depth=xgb.train$bestTune$max_depth, 
                eta=xgb.train$bestTune$eta, 
                nthread=4, 
                min_child_weight=xgb.train$bestTune$min_child_weight,
                scale_pos_weight=sumwneg/sumwpos, 
                eval_metric="auc", 
                nrounds=xgb.crv$best_iteration, 
                objective="binary:logistic")

# saveRDS(xgb.rfe2,"xgbrfe2.rds")
result.predicted1 <- predict(xgb.rfe2, newdata = dtest, type = "prob")

result.roc1 <- roc(ts_label, result.predicted1)
result.roc1
ci.auc(as.vector(ts_label), as.vector(result.predicted1))
# 95% CI: 0.8352-0.9017 (DeLong)

# Area under the curve: 0.8497

result.predicted1 <- ifelse (result.predicted1 > 0.5,1,0)
result.predicted1 <- as.factor(result.predicted1)

cm_all_1 <- confusionMatrix (result.predicted1,
                             as.factor(ts_label))
cm_all_1

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1
# 0   65  184
# 1   34 1008
# 
# Accuracy : 0.8311          
# 95% CI : (0.8096, 0.8512)
# No Information Rate : 0.9233          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.2963          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.65657         
#             Specificity : 0.84564         
#          Pos Pred Value : 0.26104         
#          Neg Pred Value : 0.96737         
#              Prevalence : 0.07668         
#          Detection Rate : 0.05035         
#    Detection Prevalence : 0.19287         
#       Balanced Accuracy : 0.75110         
#                                           
#        'Positive' Class : 0    

###############################################################################################
##############################################################################################
########## TESTING THE MODEL USING DATASET FROM COMPLETE CASES ###############
########### TESTING STEMI 1 ##############
# saveRDS(xgb.rfe1,"xgbrfe1.rds")
model = readRDS("xgbrfe1.rds")

####### TESTING STEMI RFE 1 #################
stemi <- read_csv("rfetesting.df - stemi.csv")
stemi <- subset(stemi, select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",    
                                  "ldlc","tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
                                  "hdlc","calcantagonist","arb","antiarr","statin","ecgabnormtypestelev2",
                                  "cardiaccath","ecgabnormtypestelev1","ecgabnormlocational","ecgabnormtypestdep","status"))

stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

############### different method to plot roc #####################
stemipred <- predict (model,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
# Area under the curve:  0.8391

#predict
xgbooststemi = xgb.DMatrix(data=xstemi, label=ystemi)
predictionstemi = predict(model, newdata = xgbooststemi )
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
# 0  43  97
# 1  17 406
# 
# Accuracy : 0.7975        
# 95% CI : (0.7619, 0.83)
# No Information Rate : 0.8934        
# P-Value [Acc > NIR] : 1             
# 
# Kappa : 0.33          
# 
# Mcnemar's Test P-Value : 1.372e-13     
#                                         
#             Sensitivity : 0.71667       
#             Specificity : 0.80716       
#          Pos Pred Value : 0.30714       
#          Neg Pred Value : 0.95981       
#              Prevalence : 0.10657       
#          Detection Rate : 0.07638       
#    Detection Prevalence : 0.24867       
#       Balanced Accuracy : 0.76191       
#                                         
#        'Positive' Class : 0  

########## Testing NSTEMI RFE 1######
# saveRDS(xgb.rfe1,"xgbrfe1.rds")
model = readRDS("xgbrfe1.rds")
nstemi <- read_csv("rfetesting.df - nstemi.csv")
nstemi <- subset(nstemi, select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",    
                                   "ldlc","tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
                                   "hdlc","calcantagonist","arb","antiarr","statin","ecgabnormtypestelev2",
                                   "cardiaccath","ecgabnormtypestelev1","ecgabnormlocational","ecgabnormtypestdep","status"))

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) 
nstemipred <- predict (model,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
# Area under the curve: 0.846

#predict
xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)
predictionnstemi = predict(model, newdata = xgboostnstemi )
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
# 0  43  97
# 1  17 406
# 
# Accuracy : 0.7975        
# 95% CI : (0.7619, 0.83)
# No Information Rate : 0.8934        
# P-Value [Acc > NIR] : 1             
# 
# Kappa : 0.33          
# 
# Mcnemar's Test P-Value : 1.372e-13     
#                                         
#             Sensitivity : 0.71667       
#             Specificity : 0.80716       
#          Pos Pred Value : 0.30714       
#          Neg Pred Value : 0.95981       
#              Prevalence : 0.10657       
#          Detection Rate : 0.07638       
#    Detection Prevalence : 0.24867       
#       Balanced Accuracy : 0.76191       
#                                         
#        'Positive' Class : 0    






################################################################################################
#############################################################################################
############# TESTING RFE 2 #############|
# saveRDS(xgb.rfe2,"xgbrfe2.rds")
model = readRDS("xgbrfe2.rds")

####### TESTING STEMI RFE 2 #################
stemi <- read_csv("rfetesting.df - stemi.csv")
stemi <- subset(stemi, select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",
                                  "tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
                                  "ldlc","hdlc","antiarr","statin","ecgabnormtypestelev2",
                                  "cardiaccath","status"))

stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

############### different method to plot roc #####################
stemipred <- predict (model,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
# Area under the curve:  0.8416

#predict
xgbooststemi = xgb.DMatrix(data=xstemi, label=ystemi)
predictionstemi = predict(model, newdata = xgbooststemi )
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
# 0  42 102
# 1  18 401
# 
# Accuracy : 0.7869        
# 95% CI : (0.7507, 0.82)
# No Information Rate : 0.8934        
# P-Value [Acc > NIR] : 1             
# 
# Kappa : 0.3076        
# 
# Mcnemar's Test P-Value : 3.541e-14     
#                                         
#             Sensitivity : 0.7000        
#             Specificity : 0.7972        
#          Pos Pred Value : 0.2917        
#          Neg Pred Value : 0.9570        
#              Prevalence : 0.1066        
#          Detection Rate : 0.0746        
#    Detection Prevalence : 0.2558        
#       Balanced Accuracy : 0.7486        
#                                         
#        'Positive' Class : 0    

########## Testing NSTEMI RFE 2######
# saveRDS(xgb.rfe2,"xgbrfe2.rds")
model = readRDS("xgbrfe2.rds")
nstemi <- read_csv("rfetesting.df - nstemi.csv")
nstemi <- subset(nstemi, select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",
                                   "tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
                                   "ldlc","hdlc","antiarr","statin","ecgabnormtypestelev2",
                                   "cardiaccath","status"))

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) 
nstemipred <- predict (model,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
# Area under the curve: 0.8436

#predict
xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)
predictionnstemi = predict(model, newdata = xgboostnstemi )
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
# 0  23  82
# 1  16 607
# 
# Accuracy : 0.8654          
# 95% CI : (0.8384, 0.8893)
# No Information Rate : 0.9464          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.2618          
# 
# Mcnemar's Test P-Value : 5.169e-11       
#                                           
#             Sensitivity : 0.58974         
#             Specificity : 0.88099         
#          Pos Pred Value : 0.21905         
#          Neg Pred Value : 0.97432         
#              Prevalence : 0.05357         
#          Detection Rate : 0.03159         
#    Detection Prevalence : 0.14423         
#       Balanced Accuracy : 0.73537         
#                                           
#        'Positive' Class : 0 
 

###########################################################################################################################################
#########################################################################################################################################

########## TESTING THE MODEL USING DATASET NOT COMPLETE CASES ###############
########### TESTING STEMI 1 ##############
# saveRDS(xgb.rfe1,"xgbrfe1.rds")
model = readRDS("xgbrfe1.rds")

####### TESTING STEMI RFE 1 USING DATASET NOT COMPLETE CASES#################
stemi <- read_csv("testdatasetelderly - stemi.csv")
stemi <- subset(stemi, select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",    
                                  "ldlc","tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
                                  "hdlc","calcantagonist","arb","antiarr","statin","ecgabnormtypestelev2",
                                  "cardiaccath","ecgabnormtypestelev1","ecgabnormlocational","ecgabnormtypestdep","status"))

stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

############### different method to plot roc #####################
stemipred <- predict (model,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
# Area under the curve:  0.8866

#predict
xgbooststemi = xgb.DMatrix(data=xstemi, label=ystemi)
predictionstemi = predict(model, newdata = xgbooststemi )
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
# 0  57 100
# 1  10 425
# 
# Accuracy : 0.8142          
# 95% CI : (0.7805, 0.8447)
# No Information Rate : 0.8868          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.4163          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.85075         
#             Specificity : 0.80952         
#          Pos Pred Value : 0.36306         
#          Neg Pred Value : 0.97701         
#              Prevalence : 0.11318         
#          Detection Rate : 0.09628         
#    Detection Prevalence : 0.26520         
#       Balanced Accuracy : 0.83014         
#                                           
#        'Positive' Class : 0   


########## Testing NSTEMI RFE 1 USING DATASET NOT COMPLETE CASES######
# saveRDS(xgb.rfe1,"xgbrfe1.rds")
model = readRDS("xgbrfe1.rds")
nstemi <- read_csv("testdatasetelderly - nstemi.csv")
nstemi <- subset(nstemi, select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",    
                                   "ldlc","tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
                                   "hdlc","calcantagonist","arb","antiarr","statin","ecgabnormtypestelev2",
                                   "cardiaccath","ecgabnormtypestelev1","ecgabnormlocational","ecgabnormtypestdep","status"))

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) 
nstemipred <- predict (model,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
# Area under the curve: 0.9085

#predict
xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)
predictionnstemi = predict(model, newdata = xgboostnstemi )
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
# 0  29  79
# 1  10 581
# 
# Accuracy : 0.8727          
# 95% CI : (0.8457, 0.8965)
# No Information Rate : 0.9442          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3405          
# 
# Mcnemar's Test P-Value : 5.679e-13       
#                                           
#             Sensitivity : 0.74359         
#             Specificity : 0.88030         
#          Pos Pred Value : 0.26852         
#          Neg Pred Value : 0.98308         
#              Prevalence : 0.05579         
#          Detection Rate : 0.04149         
#    Detection Prevalence : 0.15451         
#       Balanced Accuracy : 0.81195         
#                                           
#        'Positive' Class : 0   





################################################################################################
#############################################################################################
############# TESTING RFE 2 USING DATASET NOT COMPLETE CASES #############|
# saveRDS(xgb.rfe2,"xgbrfe2.rds")
model = readRDS("xgbrfe2.rds")

####### TESTING STEMI RFE 2 USING DATASET NOT COMPLETE CASES #################
stemi <- read_csv("testdatasetelderly - stemi.csv")
stemi <- subset(stemi, select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",
                                  "tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
                                  "ldlc","hdlc","antiarr","statin","ecgabnormtypestelev2",
                                  "cardiaccath","status"))

stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

############### different method to plot roc #####################
stemipred <- predict (model,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
# Area under the curve:  0.8769

ci.auc(as.vector(ystemi), as.vector(stemipred))
# 95% CI: 0.8365-0.9174 (DeLong)

#predict
xgbooststemi = xgb.DMatrix(data=xstemi, label=ystemi)
predictionstemi = predict(model, newdata = xgbooststemi )
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
# 0  56 108
# 1  11 417
# 
# Accuracy : 0.799           
# 95% CI : (0.7644, 0.8306)
# No Information Rate : 0.8868          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3862          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.83582         
#             Specificity : 0.79429         
#          Pos Pred Value : 0.34146         
#          Neg Pred Value : 0.97430         
#              Prevalence : 0.11318         
#          Detection Rate : 0.09459         
#    Detection Prevalence : 0.27703         
#       Balanced Accuracy : 0.81505         
#                                           
#        'Positive' Class : 0  

########## Testing NSTEMI RFE 2 USING DATASET NOT COMPLETE CASES######
# saveRDS(xgb.rfe2,"xgbrfe2.rds")
model = readRDS("xgbrfe2.rds")
nstemi <- read_csv("testdatasetelderly - nstemi.csv")
nstemi <- subset(nstemi, select = c("killipclass","fbg","bpsys","oralhypogly","bb","heartrate","ck",
                                   "tc","acei","crenal","bpdias","ptageatnotification","cdm","insulin",
                                   "ldlc","hdlc","antiarr","statin","ecgabnormtypestelev2",
                                   "cardiaccath","status"))

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) 
nstemipred <- predict (model,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
# Area under the curve: 0.8965

ci.auc(as.vector(ynstemi), as.vector(nstemipred))
# 95% CI: 0.847-0.9459 (DeLong)

#predict
xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)
predictionnstemi = predict(model, newdata = xgboostnstemi )
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
# 0  28  76
# 1  11 584
# 
# Accuracy : 0.8755          
# 95% CI : (0.8488, 0.8991)
# No Information Rate : 0.9442          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3379          
# 
# Mcnemar's Test P-Value : 6.813e-12       
#                                           
#             Sensitivity : 0.71795         
#             Specificity : 0.88485         
#          Pos Pred Value : 0.26923         
#          Neg Pred Value : 0.98151         
#              Prevalence : 0.05579         
#          Detection Rate : 0.04006         
#    Detection Prevalence : 0.14878         
#       Balanced Accuracy : 0.80140         
#                                           
#        'Positive' Class : 0               
#                               


