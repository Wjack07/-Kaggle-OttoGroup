
require(xgboost)
require(methods)
library(randomForest)
set.seed(342)

MultiLogLoss <- function(act, pred)
{
  eps = 1e-15;
  nr <- nrow(pred)
  pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
  pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
  ll = sum(act*log(pred))    
  ll = ll * -1/(nrow(act))      
  return(ll);
}

setwd(".../Kaggle")


LR_pred<-read.csv("LR_pred_4train.csv")
SVM_pred<-read.csv("SVM_pred_4train.csv")
XGB_pred<-read.csv("XGB_pred_4train.csv")
RF_pred<-read.csv("RF_pred_4train.csv")
CalRF_pred<-read.csv("CalRF_pred_4train.csv")

LR=matrix(unlist(LR_pred[,2:10]),nrow(LR_pred),9)
SVM=matrix(unlist(SVM_pred[,2:10]),nrow(LR_pred),9)
RF=matrix(unlist(RF_pred[,2:10]),nrow(LR_pred),9)
CalRF=matrix(unlist(CalRF_pred[,2:10]),nrow(LR_pred),9)
XGB=matrix(unlist(XGB_pred[,2:10]),nrow(XGB_pred),9)

train_buffer = read.csv("train.csv",header=TRUE,stringsAsFactors = F)
id_train=train_buffer[1]
train_y_buffer = train_buffer[95]
train_y = train_y_buffer[1:nrow(train_y_buffer),]
train = train_buffer[,-1]    ## remove the 'id' col

valid_buffer = read.csv("Train_Valid.csv",header=TRUE,stringsAsFactors = F)
id_valid=valid_buffer[1]
valid_y_buffer = valid_buffer[95]
valid_y = valid_y_buffer[1:nrow(valid_y_buffer),]
valid = valid_buffer[,-1]    ## remove the 'id' col

train_x = train[,-ncol(train)]                 ## This remove 'id' column
New_train=matrix(unlist(train_x[,1:93]),nrow(train_x),93)
New_train=cbind(LR,New_train)
#New_train=cbind(RF,New_train)
#New_train=cbind(CalRF,New_train)
#New_train=cbind(New_train,XGB)



New_train_x = as.matrix(New_train)                          ## This just change the format
New_train_x = matrix(as.numeric(New_train_x),nrow(New_train_x),ncol(New_train_x)) ## This just change the format (change to numeric)

train_y = gsub('Class_','',train_y)                   ## This change the 'Class_' into numbers
train_y_fac = as.factor(train_y)                       #R's RF assums type of factor for classif
train_y_num = as.integer(train_y)-1                       #xgboost take features in [0,numOfClass)

valid_x = valid[,-ncol(valid)]                 ## This remove 'id' column
valid_x = as.matrix(valid_x)                          ## This just change the format
valid_x = matrix(as.numeric(valid_x),nrow(valid_x),ncol(valid_x)) ## This just change the format (change to numeric)

valid_y = gsub('Class_','',valid_y)                   ## This change the 'Class_' into numbers
valid_y_fac = as.factor(valid_y)                       #R's RF assums type of factor for classif
valid_y_num = as.integer(valid_y)-1                       #xgboost take features in [0,numOfClass)




mylogloss_list=0
myparam_list=0
run=0
run_max=50

while (run<run_max)
{
  ptm <- proc.time()
  run=run+1
  # Set necessary parameter
  param_xgb <- list("objective" = "multi:softprob",
                    "eval_metric" = "mlogloss",
                    "num_class" = 9,
                    "eta" = runif(1,0.01,0.8),
                    "gamma" =1,
                    "max_depth" = sample(c(3:15),1),
                    "min_child_weight" = sample(c(2:5),1),
                    "nrounds" = 500,
                    "subsample" =0.8,
                    "colsample_bytree" =0.8,
                    "verbose"=0) # Number of trees to fit
  
  clf_xgb = xgboost(param=param_xgb, data = New_train_x, label = train_y_num, nrounds=param_xgb$nrounds, verbose=param_xgb$verbose)
  
  ## This is very specific
  ## xgboost prediction generates a 1D array, having a cetain arrangement
  ## This arrangement has to be rearranged by following
  pred_prob = predict(clf_xgb,valid_x)
  pred_prob = matrix(pred_prob,9,length(pred_prob)/9)
  pred_prob = t(pred_prob)
  actual = matrix(0,nrow=nrow(pred_prob),ncol=ncol(pred_prob))
  for (x in 1:nrow(pred_prob)){
    actual[x,valid_y_num[x]+1]=1  
  }
  mylogloss <- MultiLogLoss(actual, pred_prob)
  ptm2 <- proc.time()-ptm
  message(mylogloss, " __ calculation costs ", ptm2[1])
  
  myparam_list=rbind(myparam_list,param_xgb)
  mylogloss_list= rbind(mylogloss_list,mylogloss)
}

