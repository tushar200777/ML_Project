
import csv,os,re,sys,codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression

class regression():
     def __init__(self,path=' ',rgr_opt='lr',no_of_selected_features=None):
        self.path = path
        self.rgr_opt=rgr_opt
        self.no_of_selected_features=no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features) 

# Selection of regression techniques  
     def regression_pipeline(self):    
    # AdaBoost 
        if self.rgr_opt=='ab':
            print('\n\t### AdaBoost Regression ### \n')
            be1 = LinearRegression()              
            be2 = DecisionTreeRegressor(random_state=0)
            be3= Ridge(alpha=1.0,solver='lbfgs',positive=True)             
            rgr = AdaBoostRegressor(n_estimators=100)
            rgr_parameters = {
            'rgr__base_estimator':(be2,),
            'rgr__random_state':(0,10),
            }      
    # Decision Tree
        elif self.rgr_opt=='dt':
            print('\n\t### Decision Tree ### \n')
            rgr = DecisionTreeRegressor(random_state=40) 
            rgr_parameters = {
            'rgr__criterion':('squared_error','friedman_mse','poisson'), 
            'rgr__max_features':('auto', 'sqrt', 'log2'),
            'rgr__max_depth':(10,40,45,60),
            'rgr__ccp_alpha':(0.009,0.01,0.05,0.1),
            } 
    # Ridge Regression 
        elif self.rgr_opt=='rg':
            print('\n\t### Ridge Regression ### \n')
            rgr = Ridge(alpha=1.0,positive=True) 
            rgr_parameters = {
            'rgr__solver':('auto', 'lbfgs'),
            } 
    # Linear Regression 
        elif self.rgr_opt=='lr':   
            print('\n\t### Linear Regression ### \n')
            rgr = LinearRegression()  
            rgr_parameters = {
            'rgr__positive':(True,False),
            }         
    # Random Forest 
        elif self.rgr_opt=='rf':
            print('\n\t ### Random Forest ### \n')
            rgr = RandomForestRegressor(max_features=None)
            rgr_parameters = {
            'rgr__criterion':('squared_error',),       
            'rgr__n_estimators':(300,),
            'rgr__max_depth':(300,),
            'rgr__max_features':('sqrt',),
            }          
    # Support Vector Machine  
        elif self.rgr_opt=='svr': 
            print('\n\t### SVM Regressor ### \n')
            rgr = SVR()  
            rgr_parameters = {
            'rgr__C':(0.1,),
            'rgr__kernel':('rbf',),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)        
        return rgr,rgr_parameters     
 
# Load the data 
     def get_data(self):
          
          
          # Load the training data and target labels
          training_data = pd.read_csv(file_path + 'training_data.csv')
          target_labels = pd.read_csv(file_path + 'training_data_targets.csv', names=['target'])
 
          return training_data, target_labels
          def preprocess_data(data, labels):
          # One-Hot Encoding for categorical variable 'ADDRESS'
          encoded_data = pd.get_dummies(data, columns=['ADDRESS'])

          # Filtering outliers based on the 'labels' column <= 200
          data_with_labels = pd.concat([encoded_data, labels], axis=1)
          filtered_data = data_with_labels[data_with_labels['target'] <= 200]

         # Separating features and target variable
         features = filtered_data.drop('target', axis=1)
         target = filtered_data['target']

    return features, target

# Usage
file_path = '/path/to/data/'
training_data, target_labels = load_data(file_path)
features, target = preprocess_data(training_data, target_labels)

    
# Regression using the Gold Statndard after creating it from the raw text    
     def regression(self):  
   # Get the data
        data,labels=self.get_data()
        data=np.asarray(data)
        labels = np.asarray(labels)

# Experiments using training data only during training phase (dividing it into training and validation set)
        skf = KFold(n_splits=5)
        predicted_target=[]; actual_target=[]; 
        count=0; 
        for train_index, test_index in skf.split(data,labels):
            X_train=[]; y_train=[]; X_test=[]; y_test=[]
            for item in train_index:
                X_train.append(data[item])
                y_train.append(labels[item])
            for item in test_index:
                X_test.append(data[item])
                y_test.append(labels[item])
            count+=1                
            print('Training Phase '+str(count))

            
            rgr,rgr_parameters=self.regression_pipeline()
            pipeline = Pipeline([('feature_selection', SelectKBest(f_regression, k=self.no_of_selected_features)),('rgr', rgr),])

            #HYPER-PARAMETER TUNING
            grid = GridSearchCV(pipeline,rgr_parameters,scoring='r2',cv=10)          
            grid.fit(X_train,y_train)     
            rgr= grid.best_estimator_
            
            print('\n\n The best set of parameters of the pipiline are: ')
            print(rgr)
            
            predicted=rgr.predict(X_test)  
            for item in y_test:
                actual_target.append(item)
            for item in predicted:
                predicted_target.append(item)           

    # Regression report
        print('\n\n Performance on Training Data \n')   
        mse=mean_squared_error(actual_target,predicted_target,squared=True)
        print ('\n MSE:\t'+str(mse)) 
        rmse=mean_squared_error(actual_target,predicted_target,squared=False)
        print ('\n RMSE:\t'+str(rmse))
        r2=r2_score(actual_target,predicted_target,multioutput='variance_weighted') 
        print ('\n R2-Score:\t'+str(r2))

import warnings
warnings.filterwarnings("ignore")


rgr=regression(' ', rgr_opt='dt',
                        no_of_selected_features=4)

rgr.regression()

