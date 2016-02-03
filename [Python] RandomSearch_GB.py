from __future__ import division
import sys #This is for setting some default system parameters
import os  #This is for setting the default path
import numpy as np
import pandas as pd
import random
from random import randint


from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge   ## linear regression with square
from sklearn.linear_model import SGDClassifier   ## linear regression with square
from sklearn.linear_model import ARDRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import log_loss
#from lasagne.layers import DenseLayer
#from lasagne.layers import InputLayer
#from lasagne.layers import DropoutLayer
#from lasagne.nonlinearities import softmax
##import theano
#from lasagne.updates import nesterov_momentum
#from nolearn.lasagne import NeuralNet

import time

#my_path='D:\\Google Drive\\BigData\\Kaggle\\[Case 3] Otto Group Product Classification Challenge'
my_path='C:\\Users\\jsc69\\Google Drive\\BigData\\Kaggle\\[Case 3] Otto Group Product Classification Challenge'
np.random.seed(17411)
#os.chdir('D:\\Google Drive\\BigData\\Kaggle\\[Case 3] Otto Group Product Classification Challenge')


class Train_data_set:
    def __init__(self,X_train,X_valid,y_train,y_valid):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

class Test_data_set:
    def __init__(self,X_test,ids):
        self.X_test = X_test
        self.ids    = ids


class RF_para:
    def __init__(self,n_tree,max_features):
        self.n_tree      = n_tree
        self.max_features = max_features

class SVM_para:
    def __init__(self,random_state,probability,max_iter):
        self.random_state = random_state
        self.probability  = probability
        self.max_iter     = max_iter

class GB_para:
    def __init__(self,n_estimators, learning_rate,max_depth,random_state):
        self.n_estimators       = n_estimators
        self.learning_rate      = learning_rate
        self.max_depth          = max_depth
        self.random_state       = random_state

class LR_para:
    def __init__(self,random_state):
        self.random_state = random_state        

class NN_deap_para:
    def __init__(self):
        n_tree = None
        max_feature = None

class NN_deap_para:
    def __init__(self):
        n_tree = None
        max_feature = None
        

def logloss(y_true, y_prob, epsilon=1e-15):
    """ Simple single time logloss """    
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities    
    scores =(log_loss(y_true, y_prob))
    return scores

def load_train_data(path, train_size, dataset=Train_data_set):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    dataset.X_train, dataset.X_valid, dataset.y_train, dataset.y_valid = train_test_split(X[:, 1:-1], X[:, -1], train_size = train_size,)
    print(" -- Loaded data.")
    dataset.X_train.astype(float)
    dataset.X_valid.astype(float)
    dataset.y_train.astype(str)
    dataset.y_valid.astype(str)
    return dataset

def load_test_data(path,my_test_data):    
    df = pd.read_csv(path)
    X = df.values
    my_test_data.X_test, my_test_data.ids = X[:, 1:], X[:, 0]
    my_test_data.X_test.astype(float)
    my_test_data.ids.astype(str)
    return my_test_data

def train_clf(clf,dataset=Train_data_set):
    t = time.time()
    #print(" -- Start training Classifier.")   
    clf.fit(dataset.X_train, dataset.y_train)
    y_prob = clf.predict_proba(dataset.X_valid)
    elapsed = time.time() - t 
    #print(" -- Finished training, and it took ", elapsed, " sec")

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(dataset.y_valid)
    assert (encoder.classes_ == clf.classes_).all()

    score = logloss(y_true, y_prob)
    #print(" -- Multiclass logloss on validation set: {:.4f}.".format(score))
    return clf,score
    
def calculate_y_prob(clf,my_test_data):
    y_prob = clf.predict_proba(my_test_data.X_test)
    return y_prob

def make_submission(clf, y_prob, path_test,path_sub):    
    X_test, ids = load_test_data(path_test)    
    with open(path_sub, 'w') as f:
        f.write('id,')
        #f.write(','.join())
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + list(map(str, probs.tolist())))
            f.write(probas)
            f.write('\n')
    print(" -- Wrote submission to file {}.".format(path_sub))


#def main():
print(" - Start.")
data_size=0.9
my_trai_data_path = my_path+'\\trainsmall.csv'
my_test_data_path = my_path+'\\test.csv'
my_subm_data_path = my_path+'\\submission.csv'
my_dataset = Train_data_set(None,None,None,None)  ## Initialize this class parameter
my_test_data = Test_data_set(None,None)
my_RF_para = RF_para(10,5)  ## n_tree / max_feature
my_GB_para = GB_para(10,0.05,5,0) ## n_estimators / learning_rate / max_depth / random_state
my_LR_para = LR_para(0) ## random_state
my_SVM_para= SVM_para(0,True,10000) ## random_state / probobility / iteration
my_dataset = load_train_data(my_trai_data_path,data_size, my_dataset)
my_test_data = load_test_data(my_test_data_path,my_test_data)
    
df1 = pd.read_csv(my_trai_data_path)
df2 = pd.read_csv(my_trai_data_path, nrows=0) 
    
#t = time.time()
#clf=RandomForestClassifier()
#param_grid = {'n_estimators': [100, 180, 260],
#               'max_features': [5 , 9, 13],
#               'verbose':[1]}    
#grid=GridSearchCV(clf,param_grid,verbose=1,cv=3,n_jobs=4)
#grid.fit(my_dataset.X_train, my_dataset.y_train)    
#print(grid)
#print(grid.grid_scores_)
#print(grid.best_estimator_)       
#elapsed = time.time() - t #toc
#print 'Grid search time: :',elapsed
    
#    clf=GradientBoostingClassifier()
#    param_grid = {'loss':['deviance','exponential'],
#                  'learning_rate':[0.03,0.1,0.3],
#                  'n_estimators':[100,300,1000],
#                  'max_depth':[3,5,7],
#                  'max_features':[3,5,7],
#                  'verbose':[1]}  
#    grid=GridSearchCV(clf,param_grid,verbose=1,cv=5,n_jobs=4)
#    grid.fit(my_dataset.X_train, my_dataset.y_train)    
#    print(grid)
#    print(grid.grid_scores_)
#    print(grid.best_estimator_)
#    
#    
#clf=SVC(C=1,verbose=3)
#my_dataset.X_train = preprocessing.scale(my_dataset.X_train) 
##param_grid = {'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
#param_grid = {'kernel':['linear'],
#             'verbose':[1,0]}  
#grid=GridSearchCV(clf,param_grid,verbose=1,cv=2,n_jobs=-1)
#grid.fit(my_dataset.X_train, my_dataset.y_train)    
#print(grid)
#print(grid.grid_scores_)
#print(grid.best_estimator_)
    
#    
#    
#    clf=LogisticRegression()
#    param_grid = {'solver': ['newton-cg', 'lbfgs', 'sag'],
#                  'C':[0.3,1.0,3.0]}    
#    grid=GridSearchCV(clf,param_grid,verbose=1,cv=5,n_jobs=4)
#    grid.fit(my_dataset.X_train, my_dataset.y_train)    
#    print(grid)
#    print(grid.grid_scores_)
#    print(grid.best_estimator_)
    
    
clf_RF = RandomForestClassifier(n_estimators=my_RF_para.n_tree,
                                max_features=my_RF_para.max_features,
                                verbose=1)  
clf_GB = GradientBoostingClassifier(n_estimators=my_GB_para.n_estimators, 
                                    learning_rate=my_GB_para.learning_rate, 
                                    max_depth=my_GB_para.max_depth, 
                                    random_state=my_GB_para.random_state,
                                    verbose=1)              
clf_LR = LogisticRegression(random_state=my_LR_para.random_state,verbose=1)  
clf_SVM = SVC(random_state=my_SVM_para.random_state,
              probability=my_SVM_para.probability,
              max_iter=my_SVM_para.max_iter,
              verbose=1)    
    
    
#clf_NN = NeuralNet(layers=[
#					('input', InputLayer),
#					('dropoutf', DropoutLayer),
#					('dense0', DenseLayer),
#					('dropout', DropoutLayer),
#					('dense1', DenseLayer),
#					('dropout2', DropoutLayer), 
#					('output', DenseLayer)
#				],
#                       input_shape=(None,my_dataset.X_train.shape[1]),
#                       output_num_units=2,
#				output_nonlinearity=softmax,
#				update=nesterov_momentum,
#                       verbose=1,
#                       dropoutf_p=0.15,
#                       dense0_num_units=500,
#                       dropout_p=0.25,
#                       dense1_num_units=250,
#                       dropout2_p=0.35,
#                       eval_size=0.05,
#                       update_learning_rate= 0.01,
#				update_momentum= 0.9,
#                       max_epochs= 20
#			)    
   
#    l = InputLayer(shape=(None, my_dataset.X_train.shape[1]))
#    l = DenseLayer(l, num_units=len(np.unique(my_dataset.y_train)), nonlinearity=softmax)
#    clf_NN2 = NeuralNet(l, update_learning_rate=0.01)
#    clf_NN2.fit(my_dataset.X_train, my_dataset.y_train)   

#kernels= ['linear', 'poly', 'rbf', 'sigmoid']

#clf = ExtraTreesClassifier(n_estimators=600, 
#                           max_features=20, 
#                           min_samples_split=3,
#                           max_depth=36,
#                           random_state=23)
#
##my_dataset.X_train = preprocessing.scale(my_dataset.X_train) 
#
#clf = RandomForestClassifier(n_estimators=500,
#                             max_features=36,
#                             max_depth=36,
#                             verbose=0,
#                             n_jobs=-1)

#clf = ExtraTreesClassifier(n_estimators=600, 
#                           max_features=20, 
#                           min_samples_split=3,
#                           max_depth=36,
#                           random_state=23,
#                           n_jobs=-1)

#t = time.time()   
#clf_RF              = clf
#clf_RF_Cal          = CalibratedClassifierCV(clf_RF, method='isotonic', cv=5)
#clf_RF_Cal,score    = train_clf(clf_RF_Cal,my_dataset)
#elapsed = time.time() - t #toc
#print 'cost time ',elapsed, '  isotonic calibration score is',score
#
#t = time.time()   
#clf_RF              = clf
#clf_RF_Cal          = CalibratedClassifierCV(clf_RF, method='sigmoid', cv=5)
#clf_RF_Cal,score    = train_clf(clf_RF_Cal,my_dataset)
#elapsed = time.time() - t #toc
#print 'cost time ',elapsed, '  sigmoid calibration score is',score
#
#n_trees= [100,200,300,400,500]
#max_features =[6,9,12,15,18,21]
#max_depths=[6,9,12,15,18,21,27,36,42,48]
#
#for n_tree in n_trees:
#    for max_feature in max_features:
#        for max_depth in max_depths:
#            t = time.time()   
#            clf_RF = RandomForestClassifier(n_estimators=n_tree,
#                                            max_features=max_feature,
#                                            max_depth=max_depth,
#                                            verbose=0)
#            clf_RF_Cal          = CalibratedClassifierCV(clf_RF, method='isotonic', cv=5)
#            clf_RF_Cal,score    = train_clf(clf_RF_Cal,my_dataset) 
#            elapsed = time.time() - t #toc
#            print 'reading time:',elapsed, ' [',n_tree,',',max_feature,',', max_depth, ']   score is',score
#            #print 'reading time:',elapsed, ' [', max_depth, ']   score is',score

#going=True
#start=[100,6,6]
#step=[20,2,2]
#judge=0
#while going==True:
#    score_matrix=[]
#    score_index=[]
#    paras = []
#    for x0 in [-1,0,1]:
#        for x1 in [-1,0,1]:
#            for x2 in [-1,0,1]:
#                t = time.time()   
#                n_estimator=start[0]+x0*step[0]
#                max_feature=start[1]+x1*step[1]
#                max_depth=start[2]+x2*step[2]                
#                clf_RF = RandomForestClassifier(n_estimators=n_estimator,
#                                                max_depth=max_depth,
#                                                max_features=max_feature,
#                                                n_jobs=-1)
#                clf_RF_Cal          = CalibratedClassifierCV(clf_RF, method='isotonic', cv=5)
#                clf_RF_Cal,score    = train_clf(clf_RF_Cal,my_dataset) 
#                elapsed = time.time() - t #toc
#                print 'reading time:',elapsed, ' [',n_estimator,',',max_feature,',', max_depth, ']   score is',score
#                score_index.append([x0,x1,x2])
#                score_matrix.append(score)
#                paras.append([n_estimator,max_feature,max_depth])
#
#    index=score_index[score_matrix.index(min(score_matrix))]
#    if index ==[0,0,0]:
#        going=False
#        print 'Finished'
#    else:
#        start=np.add(start,np.multiply(index,step))
#        print 'Move on, current best is ', min(score_matrix), ' paras are ', paras[score_matrix.index(min(score_matrix))]



n_iter_search =200
run=0
paras_list =[]
score_list =[]

while run<n_iter_search:
    t = time.time()      
    param_dist = {"loss":'deviance',
                  "learning_rates":randint(2,10)*0.01,
                  "n_estimators":randint(1,30)*25,
                  "max_features": randint(2,10)*3,
                  "max_depth": randint(2,10)*3,
                  "min_samples_split": randint(1,5),
                  "min_samples_leaf": randint(1,5),
                  "subsample":randint(1,10)*0.1}
    clf =  GradientBoostingClassifier(loss=param_dist['loss'],
                                      learning_rate=param_dist['learning_rates'],
                                      n_estimators=param_dist['n_estimators'],
                                      max_features=param_dist['max_features'],
                                      max_depth=param_dist['max_depth'],
                                      min_samples_split=param_dist['min_samples_split'],
                                      min_samples_leaf=param_dist['min_samples_leaf'],  
                                      subsample=param_dist['subsample']
                                      )     
    clf,score    = train_clf(clf,my_dataset) 
    paras_list.append([param_dist['learning_rates'],param_dist['n_estimators'],param_dist['max_features'],param_dist['max_depth'],param_dist['min_samples_split'],param_dist['min_samples_leaf'],param_dist['subsample']])
    score_list.append(score)
    elapsed = time.time() - t #toc
    run=run+1
    print 'reading time:',elapsed, ' score is ',score

print 'The target parameters are :', paras_list[score_list.index(min(score_list))]
                   

print(" - Finished.")
