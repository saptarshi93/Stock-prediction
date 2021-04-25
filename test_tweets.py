
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
import re
import string
from nltk import sent_tokenize

def remove_html_tag(text):
    clean = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(clean,"",str(text))
    return text

def remove_url(string):
  return re.sub(r"http\S+", "", string)

def remove_caps(string):
    caps = re.compile('[A-Z][^a-z]')
    string = re.sub(caps,"",str(string))
    string = re.sub(' +', ' ',string)
    return string



def clean_num(text):
    pattern = '[0-9]'
    caps = re.compile(pattern)
    text = re.sub(caps,"",str(text))
    return text
 


def data_cleaning(frame):
    frame['text'] = [i.replace('\n',' ').replace('$','').replace('@','').replace('#','') for i in frame.text]
    frame['text'] = [remove_html_tag(i) for i in frame.text]
    frame['text'] = [remove_url(i) for i in frame.text]
    frame['text'] = [remove_caps(i) for i in frame.text]
    frame['text'] = [clean_num(i) for i in frame.text]
    return frame
    
def plot_graph(y_test,y_pred_df):
    plt.figure(figsize=(12,5));
    plt.title('Comparison actual vs predicted')
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

sid_obj = SentimentIntensityAnalyzer() 

'''Calculate sentiment with weighted approch using vader'''  
def weight_sentiment_scores(news,with_weight): 
    count = 0
    polarity = 0
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
    weight = len(news)
    
    if with_weight:
        for i in range(len(news)):
            
            #print(i)
            if not isinstance(news[i],float):
                weight = weight - i
                sentiment_dict = sid_obj.polarity_scores(news[i][0]) 
                polarity = polarity+(sentiment_dict['compound']*weight)
                count = count+weight
                
        if count:
            return polarity/count
        else:
            return 0.0
    else:
        for i in range(len(news)):
            
            #print(i)
            if not isinstance(news[i],float):
                weight = weight - i
                sentiment_dict = sid_obj.polarity_scores(news[i][0]) 
                polarity = polarity+(sentiment_dict['compound']*weight*news[i][1])
                count = count+weight
                
        if count:
            return polarity/count
        else:
            return 0.0
        
    
'''Calculate sentiment with weighted approch using textblob'''    
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
    
'''Calculate sentiment with weighted approch using pysentiment'''   
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
    
'''update the vader lexicon with news corpus'''
def update_vader(vader,corpus):
    return vader.lexicon.update(corpus)

sid_obj = SentimentIntensityAnalyzer() 

'''Scale the scores of the words in Sentibignomics'''
for i in senti_bignomics.senti_bignomics:
    senti_bignomics.senti_bignomics[i] = 4*float(senti_bignomics.senti_bignomics[i][0])
update_vader(sid_obj,senti_bignomics.senti_bignomics)    

'''Calculate the error'''
def get_error(y_test,y_pred):
    return np.sqrt(np.mean(np.square(((y_test - y_pred) / y_test)), axis=0))*100


'''Calculate sentiment using window'''    
def window_sentiment(data,window):
    news_list = []
    news = []
     
    data['news'] = data['pol']
    for i in range(0,window+1):
        #print(data['news'].shift(i).fillna('[]'))
        data['news'] = data['news']+data['pol'].shift(i)
    
    #print(len(news_list))
    #print(data['news']/(window))*10000000000 
    return (data['news']/(window))

'''Get the prediction'''    
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


'''Sort the news by time and date'''
def sort_news_by_time(frame,col):
    print('a')
    company = {}
    new_frame = {}
    words = None
    prev = None
    for i in frame.values:
        #print(i)
        time,key,val,score=str(i[0]),str(i[1]),i[2],i[3]
        sentiment_dict = sid_obj.polarity_scores(val)
        #print(i)
        if (key>='2018-12-29' and key<='2020-10-01'):
            if prev == key and time<'21:00:00':
                print(val)
                words.append([val,score,sentiment_dict['compound']])
            else:
                if prev is not None:
                    company[prev] = words
                words = []
                prev = key
                 
                words.append([val,score,sentiment_dict['compound']])
        company[prev] = words 
    print(company.keys())
    
    new_frame['time'] = company.keys()
    new_frame['text'] = company.values()
    #print(new_frame)
    return pd.DataFrame.from_dict(new_frame).iloc[1:,:]


'''Calculate sentiment using vader without weight'''
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

def gather_proper_news(frame):
    frame['change'] = frame['close'].shift(-1) - frame['close']
    support_list = []
    non_support_list = []
    support = []
    non_support = []
    
    for i in range(len(frame['change'])):
       if frame['text'][i]!= None: 
            #print(frame['text'][i])
            for j in frame['text'][i]:
                if (frame['change'][i]>0 and j[2]>0) or (frame['change'][i]<0 and j[2]<0):
                    print(j[1])
                    support.append(j[1]*10)
                else:
                    non_support.append(j[1]*10)
        #support_list.append(support)
        #non_support_list.append(support)
    #frame['support'] = support_list 
    #frame['non_support'] = non_support_list
    fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
    axs.hist(support, bins = 20)
    fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
    axs.hist(non_support, bins = 20)
    
    
  
    # Show plot
    plt.show()
    return frame


def prepare_data(path,column):
    if 'time' not in path.columns:
        path['time'] = path['date']
        path = path.sort_values(by=['time'])
        #path = path.mask(path.eq('None')).dropna()
        path['date'] = pd.to_datetime(path['time']).dt.date
        path['time'] = pd.to_datetime(path['time']).dt.time
        path['date'] = pd.to_datetime(path['date'])
        path = path[['time','date',column,'score']]
        print(path)
        return(path)
    else:
        path = path.sort_values(by=['time'])
        #path = path.mask(path.eq('None')).dropna()
        path['date'] = pd.to_datetime(path['time']).dt.date
        path['time'] = pd.to_datetime(path['time']).dt.time
        path['date'] = pd.to_datetime(path['date'])
        path = path[['time','date',column,'score']]
        print(path)
        return(path)
        

'''Ignore this function'''
def _get_sentiment_(frame,column):
    frame['time'] = pd.to_datetime(frame['time'])
    frame['sentiment'] = [sentiment_scores(i) for i in frame[column]]
    return frame

def load_text_data(path,column):
    df = pd.read_csv(path)
    df= df.drop_duplicates()
    df = data_cleaning(df)
    df = sort_news_by_time(prepare_data(df,column),column)
    df['time'] = pd.to_datetime(df['time'])
    #print(df.columns)
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

#def update_vader(vader,corpus):
 #   return vader.lexicon.update(corpus)

def get_sentiment(frame,with_weight):
    pol_list = []
    for i in frame['text']:
        if not isinstance(i,float):
            
            pol_list.append(weight_sentiment_scores(i,with_weight))
        else:
            pol_list.append(0)
    frame['pol'] = pol_list
    return frame

def classification(frame):
    pass

def start(pop,pcp,with_weight):

    text = load_text_data(r"C:\Users\Acer\Downloads\Data\tweet_score\AMZN_tweet_score.csv",'text')
    #print(text['text'])
    price = load_price_data(r"C:\Users\Acer\Downloads\Data\csv\AMZN.csv")
    frame = merge_frame(price,text,'time')
    frame = frame.sort_values(by=['time'])
    frame = frame.set_index('time')
    frame = frame.dropna()
    #gather_proper_news(frame)
    frame = get_sentiment(frame,with_weight)
    frame = frame.tail(458)
    frame = frame.dropna()
    
    frame['rel_pol'] = window_sentiment(frame,pop)
    frame['rel_pol'] = frame['rel_pol'].fillna(0.0)
    frame = frame['2019-02-01':'2020-10-01']
    #print(frame['rel_pol'])
    frame['close_days'] = frame['close'].shift(pcp)
    
    frame = frame[['close','rel_pol','close_days']]
    #csv = frame.to_csv(r'C:\Users\Acer\Downloads\Data\test_lstm\AAPL.csv')
    frame = frame.dropna()
    #print(frame)
    #frame['rel_pol'] = frame['rel_pol']
    #print(frame['rel_pol'].tolist())
    #correlation_raph(frame)
    print(get_prediction(frame))


start(17,-1,False)