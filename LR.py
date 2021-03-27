# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 08:52:01 2020

@author: - Saptarshi Mukhopadhaya
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



def set_window(days):
    return data['close'].shift(days)
"Plot the graph"
def plot_graph(y_test,y_pred_df):
    plt.figure(figsize=(12,5));
    plt.title('Comparzison actual vs predicted')
    plt.plot(y_test)
    y_test.plot(legend=True) 
    plt.plot(y_pred_df)
    plt.show()

"Gives the the error"
def get_error(y_test,y_pred):
    return np.sqrt(np.mean(np.square(((y_test - y_pred) / y_test)), axis=0))

def get_score(model,x,y):
    return model.score(x,y)
    
"Prediction model"
def get_prediction(data,days):
    data_new = data[['time','close','volume','open']]
    data_new['time'] = pd.to_datetime(data_new['time'])
    data_new['close_days'] = set_window(days)
    data_new = data_new.set_index('time')
    data_new = data_new['2019-01-01':]
    data_new = data_new.dropna()
    
    "Specify input and output variable"
    x_data = data_new.drop(['close_days'], axis=1)
    y_data = data_new['close_days']
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,shuffle=False)
    
    lr_model=LinearRegression()
    
    lr_model.fit(x_train,y_train)
    
    y_pred=lr_model.predict(x_test)
    y_pred_df = pd.DataFrame(y_pred, index= y_test.index)
    plot_graph(y_test,y_pred_df)
    return get_error(y_test,y_pred)
   
    
count = 0
avg_error = 0
for file in os.listdir(r"C:\\Users\\Acer\\Downloads\\Data\\csv"):
    print(file)
    count = count+1
    path = r'C:\\Users\\Acer\\Downloads\\Data\\csv\\'+file
    data = pd.read_csv(path)
    avg_error = avg_error + get_prediction(data,-30)
print((avg_error/count)*100)
