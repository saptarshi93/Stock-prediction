# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:04:56 2021

@author: Saptarshi mukhopadhaya
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import senti_bignomics
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import pysentiment2 as ps


def plot_graph(y_test,y_pred_df):
    plt.figure(figsize=(12,5));
    plt.title('Comparzison actual vs predicted')
    plt.plot(y_test)
    y_test.plot(legend=True) 
    plt.plot(y_pred_df)
    plt.show()
    
def correlation_raph(frame):
    frame['change'] = frame['close_days'] -frame['close']
    frame = frame.tail(100)
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('polarity score', color=color)
    ax1.plot(frame.index, frame['rel_pol'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('change in $', color=color)  # we already handled the x-label with ax1
    ax2.plot(frame.index, frame['change'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

'''Calculate sentiment with weighted apprach using vader'''  
def weight_sentiment_scores(news): 
    count = 0
    polarity = 0
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
    weight = len(news)
    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    for i in range(len(news)):
        #print(i)
        if not isinstance(news[i],float):
            weight = weight - i
            sentiment_dict = sid_obj.polarity_scores(news[i]) 
            polarity = polarity+(sentiment_dict['compound']*weight)
            count = count+weight
            
    if count:
        return polarity/count
    else:
        return 0.0
    
def weight_sentiment_textblob(news):
    count = 0
    polarity = 0
    weight = len(news)
    for i in news:
        sen = TextBlob(i)
        if sen.sentiment.polarity != 0.0:
            polarity = polarity+(sen.sentiment.polarity*weight)
            count = count+weight
    if count:
        return polarity/count
    else:
        return 0.0
    
def weight_sentiment_pysentiment(news):
    lm = ps.LM()
    count = 0
    polarity = 0
    weight = len(news)
    for i in news:
        #print(i)
        tokens = lm.tokenize(i)
        score = lm.get_score(tokens)
        if score['Subjectivity']!= 0.0:
            #print(score['Polarity'])
            polarity = polarity+score['Polarity']*weight
            count = count+weight
    if count:
        return polarity/count
    else:
        return 0.0
    

def update_vader(vader,corpus):
    return vader.lexicon.update(corpus)

sid_obj = SentimentIntensityAnalyzer() 
    
for i in senti_bignomics.senti_bignomics:
    senti_bignomics.senti_bignomics[i] = 4*float(senti_bignomics.senti_bignomics[i][0])
update_vader(sid_obj,senti_bignomics.senti_bignomics)    

def get_error(y_test,y_pred):
    return np.sqrt(np.mean(np.square(((y_test - y_pred) / y_test)), axis=0))*100


'''Calculate sentiment using window'''    
def window_sentiment(data,window):
    news_list = []
    news = []
     
    data['news'] = data['pol']+data['pol'].shift(-1)
    for i in range(0,window+1):
        #print(data['news'].shift(i).fillna('[]'))
        data['news'] = data['news']+data['pol'].shift(i)
    
    #print(len(news_list))
    return data['news']   

''''''    
def get_prediction(data):
    data_new = data
    
    "Specify input and output variable"
    x_data = data_new.drop(['close_days'], axis=1)
    y_data = data_new['close_days']
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,shuffle=False)
    
    lr_model=LinearRegression()
    
    lr_model.fit(x_train,y_train)
    
    y_pred=lr_model.predict(x_test)
    y_pred_df = pd.DataFrame(y_pred, index= y_test.index)
    print(x_data.columns)
    plot_graph(y_test,y_pred_df)
    return get_error(y_test,y_pred)

#def get_only_key_sen(sen)


def sort_news_by_time(frame,col):
    company = {}
    new_frame = {}
    words = None
    prev = None
    for i in frame.values:
        #print(i)
        time,key,val=str(i[0]),str(i[1]),i[2]
        #print(i)
        if (key>='2018-12-29' and key<='2020-10-01'):
            if prev == key and time<='17:00:00':
                words.append(val)
            else:
                if prev is not None:
                    company[prev] = words
                words = []
                prev = key
                words.append(val)
        company[prev] = words 
    #print(company.keys())
    
    new_frame['time'] = company.keys()
    new_frame['text'] = company.values()
    return pd.DataFrame.from_dict(new_frame).iloc[1:,:]


'''Calculate sentiment using vader'''
def sentiment_scores(news): 
    count = 0
    polarity = 0
    # Create a SentimentIntensityAnalyzer object. 
    
    
    for i in news:
        
        if not isinstance(i,float):
            
            sentiment_dict = sid_obj.polarity_scores(i) 
            polarity = polarity+(sentiment_dict['compound'])
            count = count+1
            
    if count:
        return polarity/count
    else:
        return 0.0

def gather_window_news():
    pass

def prepare_data(path,column):
    path = path.sort_values(by=['time'])
    path = path.mask(path.eq('None')).dropna()
    path['date'] = pd.to_datetime(path['time']).dt.date
    path['time'] = pd.to_datetime(path['time']).dt.time
    path['date'] = pd.to_datetime(path['date'])
    path = path[['time','date',column]]
    return(path)

def _get_sentiment_(frame,column):
    frame['time'] = pd.to_datetime(frame['time'])
    frame['sentiment'] = [sentiment_scores(i) for i in frame[column]]
    return frame

def load_text_data(path,column):
    df = sort_news_by_time(prepare_data(pd.read_csv(path),column),column)
    df['time'] = pd.to_datetime(df['time'])
    #print(df.dtypes)
    return df

def merge_frame(df,data,col):
    data = data.set_index('time')
    frame = df.join(data,on=col,how = 'left')
    #frame['pol'] = frame['pol'].fillna (0)
    return frame

def load_price_data(path):
    data = pd.read_csv(path)
    data = data[['time','open','close']]
    data = data.sort_values(by=['time'])
    #data = data.where(data['time']>'2019-01-01')
    data['time'] = pd.to_datetime(data['time'])
    #data = data.set_index('time')
    #print(data.dtypes)
    data = data.dropna()
    return data

def update_vader(vader,corpus):
    return vader.lexicon.update(corpus)

def get_sentiment(frame):
    pol_list = []
    for i in frame['text']:
        if not isinstance(i,float):
            pol_list.append(sentiment_scores(i))
        else:
            pol_list.append(0)
    frame['pol'] = pol_list
    return frame

def classification(frame):
    pass

text = load_text_data(r"C:\Users\Acer\Documents\Python Scripts\ MSFT .csv",'messages')
#print(text)
price = load_price_data(r"C:\Users\Acer\Downloads\Data\csv\MSFT.csv")
frame = merge_frame(price,text,'time')
frame = frame.sort_values(by=['time'])
frame = frame.set_index('time')
frame = get_sentiment(frame)
frame = frame.tail(458)
frame['rel_pol'] = window_sentiment(frame,25)
frame['rel_pol'] = frame['rel_pol'].fillna(0.0)
frame = frame.head(441)
#print(frame['rel_pol'])
frame['close_days'] = frame['close'].shift(-1)
frame = frame[['close','rel_pol','close_days']]
frame = frame.dropna()
#correlation_raph(frame)
print(get_prediction(frame))



