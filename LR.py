# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 08:52:01 2020

@author: -
"""
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')  
import fileinput
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

for file in os.listdir(r"C:\\Users\\Acer\\Downloads\\Data\\csv"):
    print(file)
    path = r'C:\\Users\\Acer\\Downloads\\Data\\csv\\'+file
    data = pd.read_csv(path)
    #data = pd.read_csv(r'C:\Users\Acer\Downloads\Data\csv\AAL.csv')
    data_new = data[['time','close','volume','open']]
    data_new['time'] = pd.to_datetime(data_new['time'])
    
    data_new['close'] = data['close'].shift(-7)
    #data_new['close_3'] = data['close'].shift(-2)
    #data_new['close_4'] = data['close'].shift(-3)
    #data_new['close_5'] = data['close'].shift(-4)
    #data_new['close_6'] = data['close'].shift(-5)
    #data_new['close_7'] = data['close'].shift(-6)
    #data_new['close_8'] = data['close'].shift(-7)
    #data_new['close_9'] = data['close'].shift(-8)
     
     
    data_new = data_new.set_index('time')
    
    data_new = data_new['2019-01-01':]
    
   # print(data_new)
    
    data_new = data_new.dropna()
    
    
    
    x_data = data_new.drop(['close'], axis=1)
    y_data = data_new['close']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,shuffle=False)
    
    lr_model=LinearRegression()
    
    lr_model.fit(x_train,y_train)
    
    y_pred=lr_model.predict(x_test)
    y_pred_df = pd.DataFrame(y_pred, index= y_test.index)
    #y_pred_df['predicition_val'] = pd.DataFrame(y_pred, index= y_test.index)
    #y_pred_df=y_pred_df[['predicition_val']]
    lr_train = lr_model.score(x_train, y_train)
    #print(lr_train)
    lr_test = lr_model.score(x_test, y_test)
    #print(lr_test)
    #rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    print(np.correlate(y_test,y_pred_df))
    rmspe = np.sqrt(np.mean(np.square(((y_test - y_pred) / y_test)), axis=0))
    #print(rmspe)
    plt.figure(figsize=(12,5));
    plt.title('Comparzison actual vs predicted')
    plt.plot(y_test)
    y_test.plot(legend=True) 
    plt.plot(y_pred_df)
    plt.show()
