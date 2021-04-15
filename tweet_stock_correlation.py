
!pip install scipy
import pandas as pd
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scipy.stats import chi2_contingency
from scipy.stats import chi2

def correlation_raph(frame):
    #frame['change'] = frame['close_days'] -frame['close']
    frame = frame.tail(10)
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('polarity score', color=color)
    ax1.plot(frame.index, frame['pol'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('change in $', color=color)  # we already handled the x-label with ax1
    ax2.plot(frame.index, frame['change'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def merge_frame(df,data,col):
    "Merge the clean price data and the clean tweet data"
    data = data.set_index('date')
    frame = df.join(data,on=col,how = 'left')
    #frame['pol'] = frame['pol'].fillna (0)
    return frame

'''Calculate sentiment using window'''    
def window_sentiment(data,window,cond = True):
    news_list = []
    news = []
    if cond:
      data['news'] = data['label']
      #print(data['news'][100])
      for i in range(0,window+1):
        data['news'] = data['news']+data['label'].shift(i)
      
      print(data['news'][100])
      print('-'*25)
      return data 
    else:
      data['news'] = data['label']+data['label'].shift(-1)
      for i in range(0,window+1):
          #print(data['news'].shift(i).fillna('[]'))
          data['news'] = data['news']+data['label'].shift(i)
      
      #print(len(news_list))
      return data  

''''''    

def prepare_data(path,column):
    path = path.sort_values(by=['date'])
    path = path.mask(path.eq('None')).dropna()
    
    path['time'] = pd.to_datetime(path['date']).dt.time
    path['date'] = pd.to_datetime(path['date']).dt.date
    print(path.columns)
    path['date'] = pd.to_datetime(path['date'])
    path['label'] = path[['label']].apply(lambda col:pd.Categorical(col).codes)
    path = path[['time','date',column,'label']]
    return(path)

def get_sentiment(frame):
    pol = []
    for i in frame['label']:
        if 0 not in i:
            pol.append(1)
        elif 1 not in i:
            pol.append(-1)
        else:
            frequency = collections.Counter(i)
            if frequency['1'] < frequency['0']:
                 pol.append(1)
            elif frequency['0'] < frequency['1']:
                 pol.append(-1)
            else:
                 pol.append(0)
    frame['pol'] = pol
    return frame

def correlation(table):
  stat, p, dof, expected = chi2_contingency(table)
  print('dof=%d' % dof)
  print(expected)
  # interpret test-statistic
  prob = 0.95
  critical = chi2.ppf(prob, dof)
  print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
  if abs(stat) >= critical:
    print('Dependent (reject H0)')
  else:
    print('Independent (fail to reject H0)')
  # interpret p-value
  alpha = 1.0 - prob
  print('significance=%.3f, p=%.3f' % (alpha, p))
  if p <= alpha:
    print('Dependent (reject H0)')
  else:
    print('Independent (fail to reject H0)')

def load_price_data(path):
    data = pd.read_csv(path)
    data = data[['time','open','close']]
    data = data.sort_values(by=['time'])
    #data = data.where(data['time']>'2019-01-01')
    data['date'] = pd.to_datetime(data['time'])
    #data = data.set_index('time')
    #print(data.dtypes)
    data = data.dropna()
    return data

def sort_news_by_time(frame,col):
    company = {}
    label_sens = {}
    new_frame = {}
    sen_labels = []
    words = None
    prev = None
    labels = None
    for i in frame.values:
        #print(i)
        time,key,val,sen=str(i[0]),str(i[1]),i[2],i[3]
        #print(i)
        if (key>='2012-12-29' and key<='2020-10-01'):
            if (prev == key and time<='17:00:00'):
                words.append(val)
                #labels.append(sen)
                
            else:
                if prev is not None:
                    #sen_labels.append(labels)
                    company[prev] = words
                    label_sens[prev] = labels
                   
                words = []
                labels = []
                prev = key
                words.append(val)
                labels.append(sen)
    if words!= None:
        #sen_labels.append(labels)
        company[prev] = words 
        label_sens[prev] = labels
   
    new_frame['date'] = company.keys()
    new_frame['text'] = company.values()
    new_frame['label'] = label_sens.values()
    return pd.DataFrame.from_dict(new_frame)
#.iloc[1:,:]


def load_text_data(path,column):
    df = get_sentiment(window_sentiment(sort_news_by_time(prepare_data(pd.read_csv(path),column),column),2))
    print(df.columns)
    df['date'] = pd.to_datetime(df['date'])
    #print(df.dtypes)
    return df

def change_sentiment(frame,window):
    "Find the stock price change based on the window, if window is positive then it will find the difference between diff(d,d-t) if the window is negative then it will calculate the diff(d+t,d)"
    change = []
    if window<0:
      frame['a']=frame['close'].shift(window)-frame['close']
      for i in frame['a']:
          if i > 0:
              change.append(1)
          elif i<=0:
              change.append(-1)
          else:
              change.append(0)
      frame['change'] = change
      return frame
    else:
      frame['a']=frame['close']-frame['close'].shift(window)
      for i in frame['a']:
          if i > 0:
              change.append(1)
          elif i<=0:
              change.append(-1)
          else:
              change.append(0)
      frame['change'] = change
      return frame

    

text = load_text_data('/content/MSFT_tweet.csv','text')
text = text[['date','text','pol']]
price = load_price_data("/content/drive/MyDrive/MSFT.csv")
frame = merge_frame(price,text,'date')
frame = frame.sort_values(by=['date'])
frame = frame.set_index('date')
#frame = get_sentiment(frame)
frame['pol'] = frame['pol'].fillna(0.0)

frame = frame.tail(450)
frame = change_sentiment(frame,-1)
frame = frame.dropna()
correlation_raph(frame)
print(classification_report(frame['pol'],frame['change']))
print(frame)
table = pd.crosstab(frame['pol'],frame['change'])
correlation(table)
