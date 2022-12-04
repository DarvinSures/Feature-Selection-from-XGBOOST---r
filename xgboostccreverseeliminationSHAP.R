rm(list=ls())
library(mlbench)
library(caret)
library(xgboost)
library(pROC)
library(dplyr)
library(SHAPforxgboost)
library(xgboost)
library(Matrix)
library(doParallel)
# library(tidyverse)

set.seed(1234)

setwd("D:/Documents/FYP/model/xgboost/xgboost complete cases/Reverse elimination loop")

#second cycle xgboost run
set.seed(1234)
#11 variables + cardiaccath
library(readxl)
ds <- read_excel("elderlycompletecases-sbe12.xlsx")
ds

colnames(ds) = c('Age', 'DocumentedCAD', 'HeartRate', 'SystolicBloodPressure','KillipClass', 
                 'CreatineKinase','PercutaneousCoronaryIntervention', 
                 'LowMolecularWeightHeparin', 'BetaBlocker', 'ACEInhibitor', 
                 'AngiotensionIIReceptorBlocker', 'CardiacCatheterization', 'status')



train.ind=createDataPartition(ds$status, times = 1, p=0.7, list = FALSE)
training.df=ds[train.ind, ]

#save testing.df into another excel file
testing.df=ds[-train.ind, ]

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
# Area under the curve: 0.8684
ci.auc(as.vector(ts_label), as.vector(result.predicted1))
# 95% CI: 0.8352-0.9017 (DeLong)
#####################################################


result.predicted1 <- ifelse (result.predicted1 > 0.5,1,0)
result.predicted1 <- as.factor(result.predicted1)

cm_all_1 <- confusionMatrix (result.predicted1,
                             as.factor(ts_label))
cm_all_1
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0  69 194
# 1  30 998
# 
# Accuracy : 0.8265          
# 95% CI : (0.8047, 0.8468)
# No Information Rate : 0.9233          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3036          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.69697         
#             Specificity : 0.83725         
#          Pos Pred Value : 0.26236         
#          Neg Pred Value : 0.97082         
#              Prevalence : 0.07668         
#          Detection Rate : 0.05345         
#    Detection Prevalence : 0.20372         
#       Balanced Accuracy : 0.76711         
#                                           
#        'Positive' Class : 0  


#####- FUNCTIONS FOR SHAP ANALYSIS -########
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
shap_result = shap.score.rank(xgb_model = xgb.mod, 
                              X_train =dtrain,
                              shap_approx = F)

var_importance(shap_result, top_n=12)

#SHAP summary
shap_long = shap.prep(shap = shap_result,
                      X_train = dtrain, 
                      top_n = 12)
dtrain = as.matrix(dtrain)
plot.shap.summary(data_long = shap_long)

#SHAP individual plot
xgb.plot.shap(data = dtrain, # input data
              model = xgb.mod, # xgboost model
              features = names(shap_result$mean_shap_score[1:12]), # only top 10 var
              n_col = 3, # layout option
              plot_loess = T # add red line to plot
)
 #####################################################
xgb.plot.multi.trees(model = xgb.mod)
importance_matrix <- xgb.importance(model = xgb.mod)
xgb.plot.importance(importance_matrix)


# ######### shap values ######### change to xtrain
# #shap values
xgb.plot.shap(data = dtrain, model = xgb.mod, top_n = 12)
xgb.plot.shap.summary(data = dtrain, model = xgb.mod, top_n = 12)


model = xgb.mod
########################### testing
stemi <- read_csv("testdatasetelderly - stemi.csv")
stemi <- subset(stemi, select = c("ptageatnotification","ccap","heartrate","bpsys",
                                  "killipclass","ck","pci",
                                  "lmwh","bb","acei","arb","cardiaccath","status"))

colnames(stemi) = c('Age', 'DocumentedCAD', 'HeartRate', 'SystolicBloodPressure','KillipClass', 
                 'CreatineKinase','PercutaneousCoronaryIntervention', 
                 'LowMolecularWeightHeparin', 'BetaBlocker', 'ACEInhibitor', 
                 'AngiotensionIIReceptorBlocker', 'CardiacCatheterization', 'status')

stemi$DocumentedCAD <- ifelse(test = stemi$DocumentedCAD ==1, yes = "1", no = "0")
stemi$BetaBlocker <- ifelse(test = stemi$BetaBlocker ==1, yes = "1", no = "0")
stemi$ACEInhibitor<- ifelse(test = stemi$ACEInhibitor ==1, yes = "1", no = "0")
stemi$LowMolecularWeightHeparin<- ifelse(test = stemi$LowMolecularWeightHeparin ==1, yes = "1", no = "0")
stemi$CardiacCatheterization<- ifelse(test = stemi$CardiacCatheterization ==1, yes = "1", no = "0")
stemi$PercutaneousCoronaryIntervention<- ifelse(test = stemi$PercutaneousCoronaryIntervention ==1, yes = "1", no = "0")
stemi$AngiotensionIIReceptorBlocker <- ifelse(test = stemi$AngiotensionIIReceptorBlocker ==1, yes = "1", no = "0")

stemi$status <- ifelse(test = stemi$status ==1, yes = "1", no = "0")
stemi$status <- as.numeric(stemi$status)

ystemi = stemi$status
xstemi = data.matrix(stemi %>% select(-status)) #remove status

model = xgb.mod
############### different method to plot roc #####################
stemipred <- predict (model,xstemi)
result.rocstemi <- roc(ystemi, stemipred, positive = 0, type ="prob")
result.rocstemi
ci.auc(as.vector(ystemi), as.vector(stemipred))
# 95% CI: 0.7746-0.8683 (DeLong)
plot(result.rocstemi, print.thres="best")
# Area under the curve:  0.8215

#predict
xgbooststemi = xgb.DMatrix(data=xstemi, label=ystemi)
predictionstemi = predict(model, newdata = xgbooststemi )
predictionstemi = ifelse(predictionstemi > 0.5, 1, 0)

####### confusion matrix ##########
predictionstemi = as.factor(predictionstemi)
cmstemi = confusionMatrix(predictionstemi, as.factor(ystemi), positive = '0')
cmstemi
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0  11  18
# 1  56 507
# 
# Accuracy : 0.875           
# 95% CI : (0.8456, 0.9006)
# No Information Rate : 0.8868          
# P-Value [Acc > NIR] : 0.835           
# 
# Kappa : 0.1726          
# 
# Mcnemar's Test P-Value : 1.699e-05       
#                                           
#             Sensitivity : 0.16418         
#             Specificity : 0.96571         
#          Pos Pred Value : 0.37931         
#          Neg Pred Value : 0.90053         
#              Prevalence : 0.11318         
#          Detection Rate : 0.01858         
#    Detection Prevalence : 0.04899         
#       Balanced Accuracy : 0.56495         
#                                           
#        'Positive' Class : 0          

####### Testing NSTEMI second cycle                   
nstemi <- read_csv("testdatasetelderly - nstemi.csv")
nstemi <- subset(nstemi, select =c("ptageatnotification","ccap","heartrate","bpsys",
                                   "killipclass","ck","pci",
                                   "lmwh","bb","acei","arb","cardiaccath","status"))

colnames(nstemi) = c('Age', 'DocumentedCAD', 'HeartRate', 'SystolicBloodPressure','KillipClass', 
                    'CreatineKinase','PercutaneousCoronaryIntervention', 
                    'LowMolecularWeightHeparin', 'BetaBlocker', 'ACEInhibitor', 
                    'AngiotensionIIReceptorBlocker', 'CardiacCatheterization', 'status')

nstemi$DocumentedCAD <- ifelse(test = nstemi$DocumentedCAD ==1, yes = "1", no = "0")
nstemi$BetaBlocker <- ifelse(test = nstemi$BetaBlocker ==1, yes = "1", no = "0")
nstemi$ACEInhibitor<- ifelse(test = nstemi$ACEInhibitor ==1, yes = "1", no = "0")
nstemi$LowMolecularWeightHeparin<- ifelse(test = nstemi$LowMolecularWeightHeparin ==1, yes = "1", no = "0")
nstemi$CardiacCatheterization<- ifelse(test = nstemi$CardiacCatheterization ==1, yes = "1", no = "0")
nstemi$PercutaneousCoronaryIntervention<- ifelse(test = nstemi$PercutaneousCoronaryIntervention ==1, yes = "1", no = "0")
nstemi$AngiotensionIIReceptorBlocker <- ifelse(test = nstemi$AngiotensionIIReceptorBlocker ==1, yes = "1", no = "0")

nstemi$status <- ifelse(test = nstemi$status ==1, yes = "1", no = "0")
nstemi$status <- as.numeric(nstemi$status)

ynstemi = nstemi$status
xnstemi = data.matrix(nstemi %>% select(-status)) 
nstemipred <- predict (model,xnstemi)
result.rocnstemi <- roc(ynstemi, nstemipred, positive = 0, type ="prob")
result.rocnstemi
ci.auc(as.vector(ynstemi), as.vector(nstemipred))
# 95% CI: 0.8018-0.9041 (DeLong)
plot(result.rocnstemi, print.thres="best")

# Area under the curve: 0.8529

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
# 0   6   2
# 1  33 658
# 
# Accuracy : 0.9499         
# 95% CI : (0.931, 0.9649)
# No Information Rate : 0.9442         
# P-Value [Acc > NIR] : 0.2876         
# 
# Kappa : 0.2409         
# 
# Mcnemar's Test P-Value : 3.959e-07      
#                                          
#             Sensitivity : 0.153846       
#             Specificity : 0.996970       
#          Pos Pred Value : 0.750000       
#          Neg Pred Value : 0.952243       
#              Prevalence : 0.055794       
#          Detection Rate : 0.008584       
#    Detection Prevalence : 0.011445       
#       Balanced Accuracy : 0.575408       
#                                          
#        'Positive' Class : 0 
