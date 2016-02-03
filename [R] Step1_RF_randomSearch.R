require(xgboost)
require(methods)
library(randomForest)
setwd(".../Kaggle")


train = read.csv("train.csv",header=TRUE,stringsAsFactors = F)
train = train[,-1]    ## remove the 'id' feature
#test = read.csv("test.csv",header=TRUE,stringsAsFactors = F)
#test = test[,-1]

MultiLogLoss <- function(act, pred)
{
  eps = 1e-15;
  nr <- nrow(pred)
  pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
  pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
  ll = sum(act*log(pred) + (1-act)*log(1-pred))
  ll = ll * -1/(nrow(act))      
  return(ll);
}


smp_size <- floor(0.8 * nrow(train))
set.seed(123)
train_ind <- sample(seq_len(nrow(train)), size = smp_size)

train_sample <- train[train_ind,]  # sample has the random ability
test_sample <- train[-train_ind,]

train_sample_x = train_sample[,-ncol(train_sample)]                  ## This remove 'id' column
train_sample_x = as.matrix(train_sample_x)                          ## This just change the format
train_sample_x = matrix(as.numeric(train_sample_x),nrow(train_sample_x),ncol(train_sample_x)) ## This just change the format (change to numeric)

train_sample_y = train_sample[,ncol(train_sample)]
train_sample_y = gsub('Class_','',train_sample_y)                   ## This change the 'Class_' into numbers
train_sample_y_fac = as.factor(train_sample_y)                       #R's RF assums type of factor for classif
train_sample_y_num = as.integer(train_sample_y)-1                       #xgboost take features in [0,numOfClass)

test_sample_x = test_sample[,-ncol(test_sample)]                  ## This remove 'id' column
test_sample_x = as.matrix(test_sample_x)                          ## This just change the format
test_sample_x = matrix(as.numeric(test_sample_x),nrow(test_sample_x),ncol(test_sample_x)) ## This just change the format (change to numeric)

test_sample_y = test_sample[,ncol(test_sample)]
test_sample_y = gsub('Class_','',test_sample_y)                   ## This change the 'Class_' into numbers
test_sample_y_fac = as.factor(test_sample_y)                       #xgboost take features in [0,numOfClass)
test_sample_y_num = as.integer(test_sample_y)-1 


# Set necessary parameter
param_xgb <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)

param_rf <- list("ntree" = 100,
              "mtry" = 5,
              "maxnodes" = 5,
              "importance" = TRUE)

# Training parameters
param_xgb["eta"] <- 0.3   # Learning rate
param_xgb["max_depth"] <- 6  # Tree depth
nround = 50     # Number of trees to fit

# Train the model
cv.nround <- 5
cv.nfold <- 3

clf_xgb = xgboost(param=param_xgb, data = train_sample_x, label = train_sample_y_num, nrounds = cv.nround)
clf_xgb.cv = xgb.cv(param=param_xgb, data = train_sample_x, label = train_sample_y_num, nfold = cv.nfold, nrounds = cv.nround)

## This is very specific
## xgboost prediction generates a 1D array, having a cetain arrangement
## This arrangement has to be rearranged by following
pred_prob2 = predict(clf_xgb,test_sample_x)
pred_prob2 = matrix(pred_prob2,9,length(pred_prob2)/9)
pred_prob2 = t(pred_prob2)
actual = matrix(0,nrow=nrow(pred_prob2),ncol=ncol(pred_prob2))
for (x in 1:nrow(pred_prob2)){
  actual[x,test_sample_y_num[x]+1]=1  
}
mylogloss <- MultiLogLoss(actual, pred_prob2)


clf_rf.cv = randomForest(param=param_rf, x = train_sample_x, y = train_sample_y_fac)
pred_prob = predict(clf_rf.cv,test_sample_x,type="prob")
actual = matrix(0,nrow=nrow(pred_prob),ncol=ncol(pred_prob))
for (x in 1:nrow(pred_prob)){
  actual[x,test_sample_y_num[x]+1]=1  
}
mylogloss <- MultiLogLoss(actual, pred_prob)

print(mylogloss)
# Make prediction
#pred = predict(bst,test_sample_x)
#pred = matrix(pred,9,length(pred)/9)
#pred = t(pred)

# Output submission
#pred = format(pred, digits=2,scientific=F) # shrink the size of submission
#pred = data.frame(1:nrow(pred),pred)
#names(pred) = c('id', paste0('Class_',1:9))
#write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)