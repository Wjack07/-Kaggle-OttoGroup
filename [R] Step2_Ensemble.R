
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

LR_pred<-read.csv("LR_pred.csv")
SVM_pred<-read.csv("SVM_pred.csv")
XGB_pred<-read.csv("XGB_pred.csv")
RF_pred<-read.csv("RF_pred.csv")
CalRF_pred<-read.csv("CalRF_pred.csv")

LR=matrix(unlist(LR_pred[,2:10]),nrow(LR_pred),9)
SVM=matrix(unlist(SVM_pred[,2:10]),nrow(LR_pred),9)
RF=matrix(unlist(RF_pred[,2:10]),nrow(LR_pred),9)
CalRF=matrix(unlist(CalRF_pred[,2:10]),nrow(LR_pred),9)
XGB=matrix(unlist(XGB_pred[,2:10]),nrow(LR_pred),9)

test = read.csv("train.csv",header=TRUE,stringsAsFactors = F)
test_y = test[,ncol(test)]
test_y = gsub('Class_','',test_y)                   ## This change the 'Class_' into numbers
test_y_fac = as.factor(test_y)                       #xgboost take features in [0,numOfClass)
test_y_num = as.integer(test_y) 

actual = matrix(0,nrow=nrow(LR_pred),ncol=ncol(LR_pred)-1)
for (x in 1:nrow(LR_pred)){
  actual[x,test_y_num[x]]=1  
}





mylogloss_list=50
myparam_list=0
run=0
run_max=30
range=2
while (run<run_max)
{ run=run+1
  param <- list("a1" = 0, # LR   0.6735
                "a2" = 0,
                "b1" = 0.9206726, # SVM  0.6536
                "b2" = 1.955423,
                "c1" = 1.961534, # RF   0.5588
                "c2" = 1.717742,
                "d1" = 1.815579, # CalRF 0.537
                "d2" = 1.052209,
                "e1" = runif(1,0,10), # XGB   0.46
                "e2" = runif(1,0,2)) # Number of trees to fit

  New_pred = param$a1*LR^param$a2+param$b1*SVM^param$d2+param$c1*RF^param$c2+ param$d1*CalRF^param$d2+param$e1*XGB^param$e2
  Weight = rowSums(New_pred,2)
  New_pred_cal=New_pred/Weight

  mylogloss <- MultiLogLoss(actual, New_pred_cal)
  message(mylogloss)
  myparam_list=rbind(myparam_list,param)
  mylogloss_list= rbind(mylogloss_list,mylogloss)
}
message(min(mylogloss_list))
which.min(mylogloss_list)
myparam_list[which.min(mylogloss_list),]

# Best ~0.50
