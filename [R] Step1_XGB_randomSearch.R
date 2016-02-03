require(xgboost)
require(methods)
library(randomForest)

setwd(".../Kaggle")

train = read.csv("train.csv",header=TRUE,stringsAsFactors = F)
test = read.csv("train.csv",header=TRUE,stringsAsFactors = F)
id_train=train[1]
id_test=test[1]
train = train[,-1]    ## remove the 'id' col
test = test[,-1]      ## remove the 'id' col

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

train_x = train[,-ncol(train)]                 ## This remove 'id' column
train_x = as.matrix(train_x)                          ## This just change the format
train_x = matrix(as.numeric(train_x),nrow(train_x),ncol(train_x)) ## This just change the format (change to numeric)

train_y = train[,ncol(train)]
train_y = gsub('Class_','',train_y)                   ## This change the 'Class_' into numbers
train_y_fac = as.factor(train_y)                       #R's RF assums type of factor for classif
train_y_num = as.integer(train_y)-1                       #xgboost take features in [0,numOfClass)

test_x = test[,-ncol(test)]                  ## This remove 'id' column
test_x = as.matrix(test_x)                          ## This just change the format
test_x = matrix(as.numeric(test_x),nrow(test_x),ncol(test_x)) ## This just change the format (change to numeric)

test_y = test[,ncol(test)]
test_y = gsub('Class_','',test_y)                   ## This change the 'Class_' into numbers
test_y_fac = as.factor(test_y)                       #xgboost take features in [0,numOfClass)
test_y_num = as.integer(test_y)-1 

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

  clf_xgb = xgboost(param=param_xgb, data = train_x, label = train_y_num, nrounds=param_xgb$nrounds, verbose=param_xgb$verbose)

## This is very specific
## xgboost prediction generates a 1D array, having a cetain arrangement
## This arrangement has to be rearranged by following
  pred_prob = predict(clf_xgb,test_x)
  pred_prob = matrix(pred_prob,9,length(pred_prob)/9)
  pred_prob = t(pred_prob)
  actual = matrix(0,nrow=nrow(pred_prob),ncol=ncol(pred_prob))
  for (x in 1:nrow(pred_prob)){
    actual[x,test_y_num[x]+1]=1  
  }
  mylogloss <- MultiLogLoss(actual, pred_prob)
  ptm2 <- proc.time()-ptm
  message(mylogloss, " __ calculation costs ", ptm2[1])

  myparam_list=rbind(myparam_list,param_xgb)
  mylogloss_list= rbind(mylogloss_list,mylogloss)
}



#ToWrite<-cbind(id,train)
#write.csv(train, file = "MyData.csv",row.names=1)



# Make prediction
#pred = predict(bst,test_sample_x)
#pred = matrix(pred,9,length(pred)/9)
#pred = t(pred)

# Output submission
#pred = format(pred, digits=2,scientific=F) # shrink the size of submission
#pred = data.frame(1:nrow(pred),pred)
#names(pred) = c('id', paste0('Class_',1:9))
#write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)