library(h2o)


setwd(".../Kaggle")

train = read.csv("train.csv",header=TRUE,stringsAsFactors = F)
test = read.csv("test.csv",header=TRUE,stringsAsFactors = F)
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
train_y_num = as.integer(train_y)                       #xgboost take features in [0,numOfClass)


train_y_matrix = matrix(0,nrow=nrow(train),ncol=9)
for (x in 1:nrow(train)){
  train_y_matrix[x,train_y_num[x]]=1  
}

train <- cbind(train_x,train_y_matrix)

test_x = test[,-ncol(test)]                  ## This remove 'id' column
test_x = as.matrix(test_x)                          ## This just change the format
test_x = matrix(as.numeric(test_x),nrow(test_x),ncol(test_x)) ## This just change the format (change to numeric)

test_y = test[,ncol(test)]
test_y = gsub('Class_','',test_y)                   ## This change the 'Class_' into numbers
test_y_fac = as.factor(test_y)                       #xgboost take features in [0,numOfClass)
test_y_num = as.integer(test_y) 

test_y_matrix = matrix(0,nrow=nrow(test),ncol=9)
for (x in 1:nrow(test)){
  test_y_matrix[x,test_y_num[x]]=1  
}

test<- cbind(test_x,test_y_matrix)

# Launch h2o on localhost, using all cores
localH2O = h2o.init()
h2oServer <- h2o.init(nthreads = -1)

train.hex <- as.h2o(object=train)
dim(train.hex)
test.hex <- as.h2o(object=test)
dim(test.hex)
######################################################################
### load data sets and create train/validation split
######################################################################


######################################################################
### parameter tuning with random search
######################################################################
mylogloss_list=0
myparam_list=0
models <- c()
pred <- 0
for (i in 1:9) {
  ptm <- proc.time()
  
  rand_activation <- c("RectifierWithDropout", "MaxoutWithDropout")[sample(1:2,1)]
  rand_numlayers <- sample(2:3,1)
  rand_hidden <- c(sample(90:800, rand_numlayers, T))
  rand_l1 <- runif(1, 0, 1e-3)
  rand_l2 <- runif(1, 0, 1e-2)
  rand_hidden_dropout <- c(runif(rand_numlayers, 0, 0.5))
  rand_input_dropout <- runif(1, 0, 0.5)
  rand_rho <- runif(1, 0.9, 0.999)
  rand_epsilon <- runif(1, 1e-10, 1e-4)
  rand_rate <- runif(1, 0.005, 0.02)
  rand_rate_decay <- runif(1, 0, 0.66)
  rand_momentum <- runif(1, 0, 0.5)
  rand_momentum_ramp <- runif(1, 1e-7, 1e-5)
  dlmodel <- h2o.deeplearning(x = 1:93,
                              y = 93+i, 
                              training_frame = train.hex, 
                              validation_frame = test.hex,
                              rho = rand_rho, epsilon = rand_epsilon, 
                              rate = rand_rate,
                              rate_decay = rand_rate_decay, 
                              nesterov_accelerated_gradient = T, 
                              momentum_start = rand_momentum,
                              momentum_ramp = rand_momentum_ramp,
                              activation = rand_activation, 
                              hidden = rand_hidden, 
                              l1 = rand_l1, 
                              l2 = rand_l2,
                              input_dropout_ratio = rand_input_dropout, 
                              hidden_dropout_ratios = rand_hidden_dropout, 
                              epochs = 20
  )
  models <- c(models, dlmodel)
  
  pred_buffer <- as.data.frame(h2o.predict(dlmodel, train.hex))
  
  #pred <- as.data.frame(h2o.predict(dlmodel, test.hex))
  #pred_buffer <-round(pred$predict)
  
  actual = matrix(0,nrow=nrow(train),ncol=9)
  for (x in 1:nrow(train)){
    actual[x,train_y_num[x]]=1  
  }
  
  
  pred_matrix = matrix(0,nrow=nrow(actual),ncol=ncol(actual))
  for (x in 1:nrow(actual)){
    if ((pred_buffer[x]<10) && (pred_buffer[x]>0))
    {
      pred_matrix[x,pred_buffer[x]]=1  
    }
  }
  
  mylogloss <- MultiLogLoss(actual, pred_matrix)
  ptm2 <- proc.time()-ptm
  message(mylogloss, " __ calculation costs ", ptm2[1])
  
  #myparam_list=rbind(myparam_list,param_xgb)
  mylogloss_list= rbind(mylogloss_list,mylogloss)
}


