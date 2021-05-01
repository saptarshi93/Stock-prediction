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
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import pysentiment2 as ps
from nltk import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import senti_bignomics
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import pysentiment2 as ps
from nltk import sent_tokenize
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.suppress(label_encoder_deprecation_msg, UserWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import spacy
import re



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

'''Calculate sentiment with weighted approch using vader'''  
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
            #sens = sent_tokenize(news[i])
            #for j in sens:
                #if comp.lower() in j.lower():
            sentiment_dict = sid_obj.polarity_scores(news[i]) 
            polarity = polarity+(sentiment_dict['compound']*weight)
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
def window_sentiment_news(data,window):
    news_list = []
    news = []
     
    data['news'] = data['pol']
    for i in range(0,window+1):
        #print(data['news'].shift(i).fillna('[]'))
        data['news'] = data['news']+data['pol'].shift(i)
    
    #print(len(news_list))
    return data['news']/(window) 
   
def window_sentiment_tweet(data,window):
    news_list = []
    news = []
     
    data['news'] = data['tweet_pol']
    for i in range(0,window+1):
        #print(data['news'].shift(i).fillna('[]'))
        data['news'] = data['news']+data['tweet_pol'].shift(i)
    
    #print(len(news_list))
    return data['news']/(window) 

'''Get the prediction'''    
def get_prediction(data,no_news):
    data_new = data
    if no_news:
        "Specify input and output variable"
        x_data = data_new.drop(['close_days','rel_pol'], axis=1)
        y_data = data_new['close_days']
    else:
        x_data = data_new.drop(['close_days'], axis=1)
        y_data = data_new['close_days']
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,shuffle=False)
    
    lr_model=LinearRegression()
    
    lr_model.fit(x_train,y_train)
    
    y_pred=lr_model.predict(x_test)
    y_pred_df = pd.DataFrame(y_pred, index= y_test.index)
    #print(x_data.columns)
    plot_graph(y_test,y_pred_df)
    print('r2-score:',r2_score(y_test,y_pred))
    print('mean_squared_error:',mean_squared_error(y_test,y_pred))
    return get_error(y_test,y_pred)

def classification_model(df):
    #print(df.columns)
    #df = df.drop(['change'], axis=1)
    #model = SVC(C=0.9, probability=True)
    model = RandomForestClassifier()
    #tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10,norm='l2', encoding='latin-1', ngram_range=(1,4), stop_words='english')
    #print(type(tfidf.fit_transform(df.text)))
    #clean_sentences = [cleaning(i) for i in df.text]
    #features = tfidf.fit_transform(clean_sentences)
    #print(features)
    #idf =CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,2), stop_words = 'english')
    #features = idf.fit_transform(df.text)
    #df1 = pd.DataFrame(features.toarray(), columns=tfidf.get_feature_names())
    #print(df1)
    #for i in df1.columns:
     #   df[i] = df1[i]
    #df['score'] = [sentiment_scores(i) for i in df.text]
    #df['pol_score'] = [sentiment_textblob(i) for i in df.text]
    #print(df)
    #df = df.fillna(0)
    #print(df)
    #df = df.drop(['text'], axis=1)
    
    labels = df.change
    
    #print(labels)
    df = df.drop(['change'], axis=1)
    df = df[['close','rel_pol','rel_tweet_pol','confidence']]
    #print(df.columns)
    #df = df.drop(['score','pol_score','open','close','volume'], axis=1)
    #df = df[['open','close','volume']]
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(df, labels, df.index, test_size=0.30, random_state=0,shuffle=False)
    #print(y_train.value_counts(normalize = False))
    #oversample = RandomOverSampler(sampling_strategy=0.9)
    # fit and apply the transform
    #X_train, y_train = oversample.fit_resample(X_train, y_train)
    #steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())]
    #pipeline = Pipeline(steps=steps)
    #print(X_train)
    model.fit(X_train, y_train)
    #cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
    #scores = cross_val_score(pipeline, X_train, y_train, scoring='f1_micro', cv=cv, n_jobs=-1)
    #score = mean(scores)
    # Make Predictions
    #y_pred_proba = model.predict_proba(X_test)
    #print(y_test)
    y_pred = model.predict(X_test)
    print('Classification report:\n ',classification_report(y_test,y_pred))
    print('confusion_matrix:\n',confusion_matrix(y_test,y_pred))
    #print(pd.crosstab(y_test,y_pred))



'''Sort the news by time and date'''
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
            if prev == key and time<'21:00:00':
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

def threshold(news,tweets):
    conf = []
    for i in range(len(news)):
        if (news[i]>0 and tweets[i]>0) or (news[i]<0 and tweets[i]<0):
           conf.append(news[i]+tweets[i])
        else:
           conf.append(0)
    return conf        

def gather_window_news():
    pass

def avg_news(df,pop):
    period = len(df)
    total_news = 0
    for i in df['text']:
        if not isinstance(i, float):
            total_news = total_news+len(i)
    buckets = period/pop
    return total_news/buckets
    
def prepare_data(path,column):
    path = path.sort_values(by=['time'])
    path = path.mask(path.eq('None')).dropna()
    path = path.drop_duplicates()
    path['date'] = pd.to_datetime(path['time']).dt.date
    path['time'] = pd.to_datetime(path['time']).dt.time
    path['date'] = pd.to_datetime(path['date'])
    path = path[['time','date',column]]
    return(path)

'''Ignore this function'''
def _get_sentiment_(frame,column):
    frame['time'] = pd.to_datetime(frame['time'])
    frame['sentiment'] = [sentiment_scores(i) for i in frame[column]]
    return frame

def load_text_data_combined(path,column):
    df = pd.read_csv(path[0])
    df1 = pd.read_csv(path[1])
    #print(df1.columns)
    if 'time' not in df1.columns:
        df1['time'] = df1['date']
        df1 = df1[['time','text']]
    else:
        df1 = df1[['time','text']]
    df['text'] = df[column]
    df = df[['time','text']] 
    df = pd.concat([df,df1])
    df = df.sort_values(by=['time'])
    df = sort_news_by_time(prepare_data(df,'text'),'text')
    df['time'] = pd.to_datetime(df['time'])
    
    #print(df.dtypes)
    return df
    
def load_text_data(path,column):
    df = pd.read_csv(path)
    if 'time' in df.columns:
        df = sort_news_by_time(prepare_data(df,column),column)
    else:
        df['time'] = df['date']
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

#def update_vader(vader,corpus):
 #   return vader.lexicon.update(corpus)

def get_sentiment(frame):
    pol_list = []
    for i in frame['news_text']:
        if not isinstance(i,float):
            pol_list.append(weight_sentiment_scores(i))
        else:
            pol_list.append(0)
    frame['pol'] = pol_list
    tweet_list = []
    for j in frame['text']:
        if not isinstance(j,float):
            tweet_list.append(weight_sentiment_scores(j))
        else:
            tweet_list.append(0)
    frame['tweet_pol'] = tweet_list   
    return frame

def change_sentiment(frame,window):
    change = []
    frame['a']=frame['close'].shift(window)-frame['close']
    #print(frame)
    for i in frame['a']:
        if i > 0:
            change.append(1)
        elif i<0:
            change.append(-1)
        else:
            change.append(0)
    frame['change'] = change
    #print(frame)
    return frame

def window_close(frame,window):
    name_list = []
    for i in range(1,window+1):
        name = 'close_'+str(i)
        name_list.append(name)
        frame[name] = frame['close'].shift(i)
    return frame,name_list   

def classification(frame):
    pass
for pop in [1,2,7,12,17,22,27,32,37,42,47]:
    text = load_text_data(r"C:\Users\Acer\Documents\Python Scripts\everything\only_stock_news_message.csv",'messages')
    text1 = load_text_data(r"C:\Users\Acer\Downloads\Data\tweet_score\NFLX_tweet_score.csv",'text')
   
    #text_tweets = load_text_data(r"C:\Users\Acer\Downloads\Data\tweet_score\AAPL_tweet_score.csv",'text')
    #print(text)
    price = load_price_data(r"C:\Users\Acer\Downloads\Data\csv\NFLX.csv")
    price,name_list = window_close(price,pop)
    frame = merge_frame(price,text,'time')
    frame['news_text'] = frame['text']
    frame = frame.drop(['text'], axis=1)
    frame = merge_frame(frame,text1,'time')
    frame = frame.sort_values(by=['time'])
    frame = frame.set_index('time')
    frame = get_sentiment(frame)
    frame = frame.tail(458)
    #print(frame['text'])
    frame['rel_pol'] = window_sentiment_news(frame,pop)
    frame['rel_tweet_pol'] = window_sentiment_tweet(frame,pop)
    frame['rel_pol'] = frame['rel_pol'].fillna(0.0)
    frame['rel_tweet_pol'] = frame['rel_tweet_pol'].fillna(0.0)
    #print('rel_pol:',frame['rel_pol'].to_list())
    #print('rel_tweet_pol:',frame['rel_tweet_pol'].to_list())
    frame = frame['2019-02-01':'2020-10-01']
    #print(frame.columns)
    print('PoP :',pop)
    #print('Avg news count within the pop:',avg_news(frame, pop))
    #frame = frame.head(441)
    #frame['rel_pol'] = frame['rel_pol']
    frame['rel_pol'] = frame['rel_pol']  / frame['rel_pol'].abs().max()*4
    frame['rel_tweet_pol'] = frame['rel_tweet_pol']  / frame['rel_tweet_pol'].abs().max()*4
    #frame['confidence'] = frame['rel_pol']+frame['rel_tweet_pol']
    frame['confidence'] = threshold(frame['rel_pol'],frame['rel_tweet_pol'])
    frame['confidence'] = frame['confidence']  / frame['confidence'].abs().max()*4
    frame = change_sentiment(frame,-1)
    frame['close_days'] = frame['close'].shift(-1)
    column_list = ['close','rel_pol','rel_tweet_pol','confidence','change']+name_list
    frame = frame[column_list]
    
    frame = frame.dropna()
    #print(frame['confidence'].tolist())
    classification_model(frame)
    #print(frame['rel_pol'].tolist())
    #correlation_raph(frame)
    #print('RMSPE:',get_prediction(frame,False))
    print('-'*25)
    

