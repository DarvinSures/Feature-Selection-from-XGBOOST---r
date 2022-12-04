#xgboost for all variables
rm(list=ls())
library(mlbench)
library(caret)
library(doParallel)
library(tidyverse)
library(xgboost)
library(pROC)
library(SHAPforxgboost)
library(Matrix)

set.seed(1234)

setwd("D:/Documents/FYP/model/xgboost/xgboost complete cases/Reverse elimination loop")

####### start loop ###########
library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")
ds <- ds[,-1] #remove patient id
ds<- subset(ds, select = -c(timiscorestemi,timiscorenstemi,acsstratum))
# 4305   55

ds$status <- ifelse(test = ds$status ==1, yes = "1", no = "0")
ds$status <- as.numeric(ds$status)

train.ind=createDataPartition(ds$status, times = 1, p=0.7, list = FALSE)
training.df=ds[train.ind, ]
# 3014   55
testing.df=ds[-train.ind, ]
# 1291   55

# x = sorted_data$X #get the rows names of all variable
x = colnames(training.df[,-55])  # remove ptoutcome

variable = c("all") #to store in roc list 
#result.roc1_rf$auc <- 0.8731

# auc from the full model
roc_list = data.frame(variable,result.roc1$auc) #list to store auc for all variables
roc_list 


# For loop
for(i in c(1:54)){ #from col 1 to 54
    print(x[i])
    train_data_sel = training.df[, !(names(training.df) %in% x[i])] 
    dim(train_data_sel)
    
    test_data_sel = testing.df[, !(names(testing.df) %in% x[i])] 
    
    set.seed(333)
    label=as.numeric(as.character(training.df$status))
    ts_label=as.numeric(as.character(testing.df$status))
    
    sumwpos=sum(label==1)
    sumwneg=sum(label==0)
    print(sumwneg/sumwpos)
    
    # dtest=sparse.model.matrix(status~.-1, data = data.frame(testing.df))
    # dtrain=sparse.model.matrix(status~.-1, training.df)

    dtest=sparse.model.matrix(status~.-1, data = data.frame(test_data_sel))
    dtrain=sparse.model.matrix(status~.-1, train_data_sel)

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
    
    
    xgb.train$bestTune
    # nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
    # 1      25         3 0.01     0                1                1       0.7
    
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
    
    xgb.mod=xgboost(data = dtrain, 
                    label = label, 
                    max.depth=xgb.train$bestTune$max_depth, 
                    eta=xgb.train$bestTune$eta, 
                    nthread=4, 
                    min_child_weight=xgb.train$bestTune$min_child_weight,
                    scale_pos_weight=sumwneg/sumwpos, 
                    eval_metric="auc", 
                    nrounds=xgb.crv$best_iteration, 
                    objective="binary:logistic")

    result.predicted1 <- predict(xgb.mod, newdata = dtest, type = "prob")

    result.roc1 <- roc(ts_label, result.predicted1)
    result.roc1$auc
    # Area under the curve: 0.8942,0.8624 #run all var again here and get this value for first row
  
    ##Plot roc 
    #sel_roc <- roc(as.vector(Y_test_label_sel), as.vector(sel_prob), positive = 0, type ="prob") 
    new_row = c(x[i],result.roc1$auc)
    roc_list = rbind(roc_list,new_row)
}

roc_list
# variable   result.roc1.auc
# 1                   all 0.894178579105761
# 2   ptageatnotification 0.887420733018057
# 3                 ptsex 0.898014294926913
# 4                ptrace 0.889355384780739
# 5         smokingstatus 0.896146818572657
# 6                  cdys 0.897705288048151
# 7                   cdm  0.89808147033534
# 8                  chpt 0.897308953138435
# 9              cpremcvd  0.89755078460877
# 10                  cmi 0.894333082545142
# 11                 ccap 0.891337059329321
# 12         canginamt2wk 0.897476891659501
# 13       canginapast2wk 0.890383168529665
# 14           cheartfail 0.891222861134996
# 15                clung 0.897329105760963
# 16               crenal 0.892008813413586
# 17     ccerebrovascular 0.889650956577816
# 18           cpvascular 0.897476891659501
# 19            heartrate 0.885781653052451
# 20                bpsys 0.887185619088564
# 21               bpdias 0.888166380051591
# 22          killipclass 0.876504729148753
# 23                   ck  0.88519722699914
# 24                   tc 0.892405148323302
# 25                 hdlc  0.88451203783319
# 26                 ldlc 0.897308953138435
# 27                   tg 0.896422237747206
# 28                  fbg 0.884874785038693
# 29 ecgabnormtypestelev1 0.889577063628547
# 30 ecgabnormtypestelev2 0.891854309974205
# 31   ecgabnormtypestdep 0.896287886930353
# 32   ecgabnormtypetwave 0.897389563628547
# 33     ecgabnormtypebbb  0.89755078460877
# 34  ecgabnormlocationil 0.897073839208942
# 35  ecgabnormlocational 0.896885748065348
# 36  ecgabnormlocationll 0.897698570507309
# 37  ecgabnormlocationtp 0.897476891659501
# 38  ecgabnormlocationrv  0.89755078460877
# 39          cardiaccath 0.895716895958727
# 40                  pci  0.89392331255374
# 41                 cabg 0.897476891659501
# 42                  asa 0.897476891659501
# 43                 gpri 0.897476891659501
# 44              heparin 0.897544067067928
# 45                 lmwh 0.895273538263113
# 46                   bb 0.883779825881341
# 47                 acei 0.885177074376612
# 48                  arb 0.885459211092003
# 49               statin 0.889476300515907
# 50              lipidla 0.897476891659501
# 51             diuretic 0.898477805245056
# 52       calcantagonist 0.897053686586414
# 53          oralhypogly 0.879883652192605
# 54              insulin 0.896852160361135
# 55              antiarr 0.895904987102322


roc_list_sort = roc_list[order(roc_list$result.roc1.auc),]
roc_list_sort

#this is after second time, lower AUC figure out why
# write.csv(roc_list, "List After Eliminating_xgboost2nd.csv")

#plot graph
library(readr)
roc_list <- read.csv("List Auc ALLVAR.csv")
#remove index col
roc_list = roc_list[,-1]
#remove all var first
auclist = roc_list[-1,]
#54
# labelsauc = c("ptageatnotification","ptsex","ptrace","smokingstatus","cdys","cdm","chpt",
#                  "cpremcvd","cmi","ccap","canginamt2wk","canginapast2wk","cheartfail","clung", 
#                  "crenal","ccerebrovascular","cpvascular","heartrate","bpsys","bpdias","killipclass",
#                  "ck","tc","hdlc","ldlc","tg","fbg","ecgabnormtypestelev1","ecgabnormtypestelev2",
#                  "ecgabnormtypestdep","ecgabnormtypetwave","ecgabnormtypebbb","ecgabnormlocationil",
#                  "ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp","ecgabnormlocationrv",
#                  "cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb","acei","arb",
#                  "statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr")

#how to add variables as labels?
plot(auclist$result.roc1.auc, type="l", xlab = "Variables", ylab = "AUC", main = "AUC for All variables")
abline(h=0.8624161, col="blue")

DF <- as.data.frame(roc_list)
firstcyclevar = DF[DF$result.roc1.auc < 0.8624161, ]
firstcyclevar$variable # take these variables only
 # (24)
# variable result.roc1.auc
# 2   ptageatnotification       0.8559589
# 7                   cdm       0.8621788
# 11                 ccap       0.8554844
# 16               crenal       0.8582469
# 19            heartrate       0.8599248
# 20                bpsys       0.8571876
# 21               bpdias       0.8608230
# 22          killipclass       0.8487136
# 23                   ck       0.8386550
# 24                   tc       0.8603230
# 25                 hdlc       0.8587130
# 27                   tg       0.8597892
# 30 ecgabnormtypestelev2       0.8623907
# 34  ecgabnormlocationil       0.8623992
# 36  ecgabnormlocationll       0.8620178
# 39          cardiaccath       0.8623737
# 40                  pci       0.8606027
# 45                 lmwh       0.8613992
# 46                   bb       0.8578656
# 47                 acei       0.8524846
# 48                  arb       0.8575690
# 49               statin       0.8588570
# 54              insulin       0.8618483
# 55              antiarr       0.8614755

####################################################################################################################
#first cycle RESULT xgboost run

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
ds <- subset(ds, select = c("ptageatnotification","cdm","ccap","crenal","heartrate","bpsys",
                             "bpdias","killipclass","ck","tc","hdlc","tg","ecgabnormtypestelev2",
                             "ecgabnormlocationil","ecgabnormlocationll","cardiaccath","pci",
                             "lmwh","bb","acei","arb","statin","insulin","antiarr","status"))

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

xgb.mod=xgboost(data = dtrain, 
                label = label, 
                max.depth=xgb.train$bestTune$max_depth, 
                eta=xgb.train$bestTune$eta, 
                nthread=4, 
                min_child_weight=xgb.train$bestTune$min_child_weight,
                scale_pos_weight=sumwneg/sumwpos, 
                eval_metric="auc", 
                nrounds=xgb.crv$best_iteration, 
                objective="binary:logistic")


result.predicted1 <- predict(xgb.mod, newdata = dtest, type = "prob")

result.roc1 <- roc(ts_label, result.predicted1)
result.roc1
# Area under the curve: 0.8842

result.predicted1 <- ifelse (result.predicted1 > 0.5,1,0)
result.predicted1 <- as.factor(result.predicted1)

cm_all_1 <- confusionMatrix (result.predicted1,
                             as.factor(ts_label))
cm_all_1
# onfusion Matrix and Statistics
# 
# Reference
# Prediction    0    1
# 0   73  168
# 1   26 1024
# 
# Accuracy : 0.8497          
# 95% CI : (0.8291, 0.8688)
# No Information Rate : 0.9233          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3598          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.73737         
#             Specificity : 0.85906         
#          Pos Pred Value : 0.30290         
#          Neg Pred Value : 0.97524         
#              Prevalence : 0.07668         
#          Detection Rate : 0.05655         
#    Detection Prevalence : 0.18668         
#       Balanced Accuracy : 0.79822         
#                                           
#        'Positive' Class : 0   


#################### second cycle #################################
#24 variables
library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")
ds <- ds[,-1] #remove patient id
ds<- subset(ds, select = -c(timiscorestemi,timiscorenstemi,acsstratum))
ds2 <- subset(ds, select = c("ptageatnotification","cdm","ccap","crenal","heartrate","bpsys",
                            "bpdias","killipclass","ck","tc","hdlc","tg","ecgabnormtypestelev2",
                            "ecgabnormlocationil","ecgabnormlocationll","cardiaccath","pci",
                            "lmwh","bb","acei","arb","statin","insulin","antiarr","status"))

ds2$status <- ifelse(test = ds2$status ==1, yes = "1", no = "0")
ds2$status <- as.numeric(ds2$status)

train.ind=createDataPartition(ds2$status, times = 1, p=0.7, list = FALSE)
training.df=ds2[train.ind, ]
# 3014   25
testing.df=ds2[-train.ind, ]
# 1291   25

# x = sorted_data$X #get the rows names of all variable
x = colnames(training.df[,-25])  # remove ptoutcome

variable = c("all") #to store in roc list 
#result.roc1_rf$auc <- 0.8731

allvar = 0.8624161
# auc from the full model
roc_list = data.frame(variable,allvar) #list to store auc for all variables
roc_list 


# For loop
for(i in c(1:24)){ #from col 1 to 54
  print(x[i])
  train_data_sel = training.df[, !(names(training.df) %in% x[i])] 
  dim(train_data_sel)
  
  test_data_sel = testing.df[, !(names(testing.df) %in% x[i])] 
  
  set.seed(333)
  label=as.numeric(as.character(training.df$status))
  ts_label=as.numeric(as.character(testing.df$status))
  
  sumwpos=sum(label==1)
  sumwneg=sum(label==0)
  print(sumwneg/sumwpos)
  
  # dtest=sparse.model.matrix(status~.-1, data = data.frame(testing.df))
  # dtrain=sparse.model.matrix(status~.-1, training.df)
  
  dtest=sparse.model.matrix(status~.-1, data = data.frame(test_data_sel))
  dtrain=sparse.model.matrix(status~.-1, train_data_sel)
  
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
  
  
  xgb.train$bestTune
  # nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
  # 1      25         3 0.01     0                1                1       0.7
  
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
  
  xgb.mod=xgboost(data = dtrain, 
                  label = label, 
                  max.depth=xgb.train$bestTune$max_depth, 
                  eta=xgb.train$bestTune$eta, 
                  nthread=4, 
                  min_child_weight=xgb.train$bestTune$min_child_weight,
                  scale_pos_weight=sumwneg/sumwpos, 
                  eval_metric="auc", 
                  nrounds=xgb.crv$best_iteration, 
                  objective="binary:logistic")
  
  result.predicted1 <- predict(xgb.mod, newdata = dtest, type = "prob")
  
  result.roc1 <- roc(ts_label, result.predicted1)
  result.roc1$auc
  
  ## Plot roc 
  #sel_roc <- roc(as.vector(Y_test_label_sel), as.vector(sel_prob), positive = 0, type ="prob") 
  new_row = c(x[i],result.roc1$auc)
  roc_list = rbind(roc_list,new_row)
}

roc_list

write.csv(roc_list, "reverseliminationsecondcycle.csv")

#plot graph 
library(readr)
roc_list <- read.csv("reverseliminationsecondcycle.csv")
#remove index col
roc_list = roc_list[,-1]
#remove all var first
auclist = roc_list[-1,]

plot(auclist$allvar, type="l", xlab = "Variables", ylab = "AUC", main = "AUC for SBE Second Cycle", pch=19)
abline(h=0.8624161, col="blue")

DF2 <- as.data.frame(roc_list)
secondcyclevar = DF2[DF2$allvar < 0.8624161, ]
secondcyclevar$variable # take these variables only

#11 variables
# [1] "ptageatnotification" "ccap"                "heartrate"           "bpsys"               "killipclass"
# [6] "ck"                  "pci"                 "lmwh"                "bb"                  "acei"
# [11] "arb"

 #####################################################################################################################
#second cycle xgboost run
set.seed(1234)
#11 variables + cardiaccath
library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")
ds <- ds[,-1]
ds <- ds %>%
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"),
            funs(recode(., `1`=1, `2`=0, .default = 0)))

ds
ds<- subset(ds, select = -c(timiscorestemi,timiscorenstemi,acsstratum))
ds <- subset(ds, select = c("ptageatnotification","ccap","heartrate","bpsys",
                             "killipclass","ck","pci",
                             "lmwh","bb","acei","arb","cardiaccath", "status"))

colnames(ds) = c('Age', 'DocumentedCAD', 'HeartRate', 'SystolicBloodPressure','KillipClass', 'CreatinineKinase', 
                 'PercutaneousCoronaryIntervention', 'LowMolecularWeightHeparin',
                     'BetaBlocker', 'ACEInhibitor', 'AngiotensionIIReceptorBlocker', 'CardiacCatheterization','status')

# ds <- setNames(ds, c('Age', 'CoronaryArteryDisorders', 'Heartrate', 'SystolicBloodBressure','Killipclass', 
#                      'CreatinineKinase', 'PercutaneousCoronaryIntervention', 'LMWHeparin', 
#                      'BetaBlocker', 'ACEInhibitor', 'AngiotensionReceptorBlocker', 
#                      'CardiacCatheterization', 'status'))

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

xgb.mod=xgboost(data = dtrain, 
                label = label, 
                max.depth=xgb.train$bestTune$max_depth, 
                eta=xgb.train$bestTune$eta, 
                nthread=4, 
                min_child_weight=xgb.train$bestTune$min_child_weight,
                scale_pos_weight=sumwneg/sumwpos, 
                eval_metric="auc", 
                nrounds=xgb.crv$best_iteration, 
                objective="binary:logistic")


result.predicted1 <- predict(xgb.mod, newdata = dtest, type = "prob")

result.roc1 <- roc(ts_label, result.predicted1)
result.roc1
# Area under the curve:0.8684

# saveRDS(xgb.mod,"xgbre2.rds")
# model = readRDS("xgbre2.rds")

data = data.frame(x = result.predicted1,
                  y = testing.df$status)

# data = data[,-1]
#first
first = data[data[,1] < 0.09 , ]
dim(first)
# 20  2

#second
second = data[data[,1] < 0.19 & data[,1] > 0.09 , ]
dim(second)
# 51  2

#third
third = data[data[,1] < 0.29 & data[,1] > 0.19 , ]
dim(third)
# 55  2

#fourth
fourth = data[data[,1] < 0.39 & data[,1] > 0.29 , ]
dim(fourth)
# 55  2

#fifth
fifth = data[data[,1] < 0.49 & data[,1] > 0.39 , ]
dim(fifth)
# 77   2

#sixth
sixth = data[data[,1] < 0.59 & data[,1] > 0.49 , ]
dim(sixth)
# 141  2

#seventh
seventh = data[data[,1] < 0.69 & data[,1] > 0.59 , ]
dim(seventh)
# 171   2

#eight
eight = data[data[,1] < 0.79 & data[,1] > 0.69 , ]
dim(eight)
# 251   2

#nine
nine = data[data[,1] < 0.89 & data[,1] > 0.79 , ]
dim(nine)
# 335    2

#tenth
tenth = data[data[,1] < 1 & data[,1] > 0.89 , ]
dim(tenth)
# 170   2

result.predicted1 <- ifelse (result.predicted1 > 0.5,1,0)
result.predicted1 <- as.factor(result.predicted1)

cm_all_1 <- confusionMatrix (result.predicted1,
                             as.factor(ts_label))
cm_all_1
# Accuracy : 0.8265          
# 95% CI : (0.8047, 0.8468)


#Testing STEMI second cycle
# model = readRDS("xgbre2.rds")
stemi <- read_csv("testdatasetelderly - stemi.csv")
stemi <- stemi %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

stemi
stemi <- subset(stemi, select = c("ptageatnotification","ccap","heartrate","bpsys",
                                  "killipclass","ck","pci",
                                  "lmwh","bb","acei","arb","cardiaccath","status"))

stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

#different method to plot roc 
stemipred <- predict (model,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
# 0.8215
ci.auc(as.vector(ystemi), as.vector(stemipred))
# 95% CI: 0.8116-0.895

#predict
xgbooststemi = xgb.DMatrix(data=xstemi, label=ystemi)
predictionstemi = predict(model, newdata = xgbooststemi )
predictionstemi = ifelse(predictionstemi > 0.5, 1, 0)
predictionstemi
table(predictionstemi)

#confusion matrix 
predictionstemi = as.factor(predictionstemi)
cmstemi = confusionMatrix(predictionstemi, as.factor(ystemi), positive = '0')
cmstemi
# Accuracy : 0.875           
# 95% CI : (0.8456, 0.9006)          

#Testing NSTEMI second cycle                   
nstemi <- read_csv("testdatasetelderly - nstemi.csv")
nstemi <- nstemi %>% 
  mutate_at(c("ptsex","cdys","cdm","chpt","cpremcvd","cmi","ccap","canginamt2wk",
              "canginapast2wk","cheartfail","clung","crenal","ccerebrovascular","cpvascular",
              "ecgabnormtypestelev1","ecgabnormtypestelev2","ecgabnormtypestdep","ecgabnormtypetwave",
              "ecgabnormtypebbb","ecgabnormlocationil","ecgabnormlocational","ecgabnormlocationll","ecgabnormlocationtp",
              "ecgabnormlocationrv","cardiaccath","pci","cabg","asa","gpri","heparin","lmwh","bb",                 
              "acei","arb","statin","lipidla","diuretic","calcantagonist","oralhypogly","insulin","antiarr","status"), 
            funs(recode(., `1`=1, `2`=0, .default = 0)))

nstemi
nstemi <- subset(nstemi, select =c("ptageatnotification","ccap","heartrate","bpsys",
                                  "killipclass","ck","pci",
                                  "lmwh","bb","acei","arb","cardiaccath","status"))

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) 
nstemipred <- predict (model,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
# 0.8529
ci.auc(as.vector(ynstemi), as.vector(nstemipred))
# 95% CI: 0.8018-0.9041 (DeLong)

#predict
xgboostnstemi = xgb.DMatrix(data=xnstemi, label=ynstemi)
predictionnstemi = predict(model, newdata = xgboostnstemi )
predictionnstemi = ifelse(predictionnstemi > 0.5, 1, 0)
predictionnstemi
table(predictionnstemi)

#confusion matrix 
predictionnstemi = as.factor(predictionnstemi) 
cmnstemi = confusionMatrix(predictionnstemi, as.factor(ynstemi), positive = '0')
cmnstemi
# Accuracy : 0.9499         
# 95% CI : (0.931, 0.9649)

############ third cycle #####################
# 11 variables + cardiaccath
library(readxl)
ds <- read_excel("elderlycompletecases.xlsx")
ds <- ds[,-1] #remove patient id
ds<- subset(ds, select = -c(timiscorestemi,timiscorenstemi,acsstratum))
ds3 <- subset(ds, select = c("ptageatnotification","ccap","heartrate","bpsys","killipclass",
                             "ck","pci","lmwh","bb","acei","arb","status"))

ds3$status <- ifelse(test = ds3$status ==1, yes = "1", no = "0")
ds3$status <- as.numeric(ds3$status)

train.ind=createDataPartition(ds3$status, times = 1, p=0.7, list = FALSE)
training.df=ds3[train.ind, ]
# 3014   12
testing.df=ds3[-train.ind, ]
# 1291   12

# x = sorted_data$X #get the rows names of all variable
x = colnames(training.df[,-12])  # remove ptoutcome

variable = c("all") #to store in roc list 
#result.roc1_rf$auc <- 0.8731

allvar = 0.8624161
# auc from the full model
roc_list = data.frame(variable,allvar) #list to store auc for all variables
roc_list 


# For loop
for(i in c(1:11)){ 
  print(x[i])
  train_data_sel = training.df[, !(names(training.df) %in% x[i])] 
  dim(train_data_sel)
  
  test_data_sel = testing.df[, !(names(testing.df) %in% x[i])] 
  
  set.seed(333)
  label=as.numeric(as.character(training.df$status))
  ts_label=as.numeric(as.character(testing.df$status))
  
  sumwpos=sum(label==1)
  sumwneg=sum(label==0)
  print(sumwneg/sumwpos)
  
  # dtest=sparse.model.matrix(status~.-1, data = data.frame(testing.df))
  # dtrain=sparse.model.matrix(status~.-1, training.df)
  
  dtest=sparse.model.matrix(status~.-1, data = data.frame(test_data_sel))
  dtrain=sparse.model.matrix(status~.-1, train_data_sel)
  
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
  
  
  xgb.train$bestTune
  # nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
  # 1      25         3 0.01     0                1                1       0.7
  
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
  
  xgb.mod=xgboost(data = dtrain, 
                  label = label, 
                  max.depth=xgb.train$bestTune$max_depth, 
                  eta=xgb.train$bestTune$eta, 
                  nthread=4, 
                  min_child_weight=xgb.train$bestTune$min_child_weight,
                  scale_pos_weight=sumwneg/sumwpos, 
                  eval_metric="auc", 
                  nrounds=xgb.crv$best_iteration, 
                  objective="binary:logistic")
  
  result.predicted1 <- predict(xgb.mod, newdata = dtest, type = "prob")
  
  result.roc1 <- roc(ts_label, result.predicted1)
  result.roc1$auc
  
  ## Plot roc 
  #sel_roc <- roc(as.vector(Y_test_label_sel), as.vector(sel_prob), positive = 0, type ="prob") 
  new_row = c(x[i],result.roc1$auc)
  roc_list = rbind(roc_list,new_row)
}

roc_list

write.csv(roc_list, "reverseliminationthirdcycle.csv")

#plot graph 
library(readr)
roc_list <- read.csv("reverseliminationthirdcycle.csv")
#remove index col
roc_list = roc_list[,-1]
#remove all var first
auclist = roc_list[-1,]

plot(auclist$allvar, type="l", xlab = "Variables", ylab = "AUC", main = "AUC for RFE Third Cycle")
abline(h=0.8624161, col="blue")

DF3 <- as.data.frame(roc_list)
thirdcyclevar = DF3[DF3$allvar < 0.8624161, ]
thirdcyclevar$variable
# "ptageatnotification" "killipclass"         "ck"  





######### shap###################
# model = readRDS("xgbre2.rds")
shap.score.rank <- function(xgb_model = xgb_mod, shap_approx = TRUE, 
                            X_train = mydata$train_mm){
  require(xgboost)
  require(data.table)
  shap_contrib <- predict(xgb_model, X_train,
                          predcontrib = TRUE, approxcontrib = shap_approx)
  shap_contrib <- as.data.table(shap_contrib)
  shap_contrib[,BIAS:=NULL]
  cat('make SHAP score by decreasing order\n\n')
  mean_shap_score <- colMeans(abs(shap_contrib))[order(colMeans(abs(shap_contrib)), decreasing = T)]
  return(list(shap_score = shap_contrib,
              mean_shap_score = (mean_shap_score)))
}

# a function to standardize feature values into same range
std1 <- function(x){
  return ((x - min(x, na.rm = T))/(max(x, na.rm = T) - min(x, na.rm = T)))
}


# prep shap data
shap.prep <- function(shap  = shap_result, X_train = mydata$train_mm, top_n){
  require(ggforce)
  # descending order
  if (missing(top_n)) top_n <- dim(X_train)[2] # by default, use all features
  if (!top_n%in%c(1:dim(X_train)[2])) stop('supply correct top_n')
  require(data.table)
  shap_score_sub <- as.data.table(shap$shap_score)
  shap_score_sub <- shap_score_sub[, names(shap$mean_shap_score)[1:top_n], with = F]
  shap_score_long <- melt.data.table(shap_score_sub, measure.vars = colnames(shap_score_sub))
  
  # feature values: the values in the original dataset
  fv_sub <- as.data.table(X_train)[, names(shap$mean_shap_score)[1:top_n], with = F]
  # standardize feature values
  fv_sub_long <- melt.data.table(fv_sub, measure.vars = colnames(fv_sub))
  fv_sub_long[, stdfvalue := std1(value), by = "variable"]
  # SHAP value: value
  # raw feature value: rfvalue; 
  # standarized: stdfvalue
  names(fv_sub_long) <- c("variable", "rfvalue", "stdfvalue" )
  shap_long2 <- cbind(shap_score_long, fv_sub_long[,c('rfvalue','stdfvalue')])
  shap_long2[, mean_value := mean(abs(value)), by = variable]
  setkey(shap_long2, variable)
  return(shap_long2) 
}

plot.shap.summary <- function(data_long){
  x_bound <- max(abs(data_long$value))
  require('ggforce') # for `geom_sina`
  plot1 <- ggplot(data = data_long)+
    coord_flip() + 
    # sina plot: 
    geom_sina(aes(x = variable, y = value, color = stdfvalue)) +
    # print the mean absolute value: 
    geom_text(data = unique(data_long[, c("variable", "mean_value"), with = F]),
              aes(x = variable, y=-Inf, label = sprintf("%.3f", mean_value)),
              size = 3, alpha = 0.7,
              hjust = -0.2, 
              fontface = "bold") + # bold
    # # add a "SHAP" bar notation
    # annotate("text", x = -Inf, y = -Inf, vjust = -0.2, hjust = 0, size = 3,
    #          label = expression(group("|", bar(SHAP), "|"))) + 
    scale_color_gradient(low="#FFCC33", high="#6600CC", 
                         breaks=c(0,1), labels=c("Low","High")) +
    theme_bw() + 
    theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), # remove axis line
          legend.position="bottom") + 
    geom_hline(yintercept = 0) + # the vertical line
    scale_y_continuous(limits = c(-x_bound, x_bound)) +
    # reverse the order of features
    scale_x_discrete(limits = rev(levels(data_long$variable)) 
    ) + 
    labs(y = "SHAP value (impact on model output)", x = "", color = "Feature value") 
  return(plot1)
}

var_importance <- function(shap_result, top_n=10)
{
  var_importance=tibble(var=names(shap_result$mean_shap_score), importance=shap_result$mean_shap_score)
  
  var_importance=var_importance[1:top_n,]
  
  ggplot(var_importance, aes(x=reorder(var,importance), y=importance)) + 
    geom_bar(stat = "identity") + 
    coord_flip() + 
    theme_light() + 
    theme(axis.title.y=element_blank()) 
}
######- END OF FUNCTION -######

#SHAP importance
shap_result = shap.score.rank(xgb_model = model, 
                              X_train =dtrain,
                              shap_approx =F)

var_importance(shap_result, top_n=12)

#SHAP summary
shap_long = shap.prep(shap = shap_result,
                      X_train = as.matrix(dtrain), 
                      top_n = 12)

plot.shap.summary(data_long = shap_long)

#SHAP individual plot
xgb.plot.shap(data = dtrain, # input data
              model = model, # xgboost model
              features = names(shap_result$mean_shap_score[1:12]), # only top 10 var
              n_col = 3, # layout option
              plot_loess = T # add red line to plot
)


