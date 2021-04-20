import pandas as pd
news_fin_list = ['buyback','putcall','put call','market momentum','dividend',' ma ',' ipo ','fundamentals','technical analysis','insider activity','volatility','buyside','price action',' vix ','high low','coverage','stock signal','stocks','stock','investment','reputation','appointment','regulatory','rumors','risks','strategy','currency','business','nasdaq','trade','tech','index','finance','stockswal','investor',' sp ','economi','econom','ipo','sale','volatil',' ceo ','profit','fundamental','insider activity','legal','forecast','analyst','revenu','profit',' loss ','buyback','putcall','put call','market momentum','central banks',' AAL ',' AMD ',' ALGN ',' ADI ',' AAPL ',' AMAT ',' ADP ',' ADSK ',' AVGO ',' BIDU ',' BIIB ',' BMRN ',' CDNS ',' CELG ',' CERN ','CTRP',' CHKP ',' CHTR ',' DLTR ',' EA ',' GOOGL ',' HSIC ',' ILMN ',' INCY ',' INTC ',' INTU ',' ISRG ',' IDXX ',' JBHT ',' JD ',' KLAC ',' KHC ',' LRCX ',' LBTYA ',' LULU ',' MELI ',' MAR ',' MCHP ',' MDLZ ',' MNST ',' MSFT ',' MU ',' MXIM ',' MYL ',' NTAP ',' NFLX ',' NTES ',' NVDA ','Nvidia','NVDA',' NXPI ',' PAYX ',' PCAR ',' BKNG ',' PYPL ',' PEP ',' QCOM ',' REGN ',' ROST ',' SIRI ',' SWKS ',' SBUX ',' SYMC ',' SNPS ',' TTWO ',' TSLA ',' TXN ',' TMUS ',' ULTA ',' UAL ',' VRSN ',' VRSK ',' VRTX ',' WBA ',' WDC ',' WDAY ',' WLTW ',' WYNN ',' XEL ',' XLNX ']

"Cleans the data,takes the data frame and the column name as argument,returns the modeified dataframe"
def data_cleaning(data,column):
    kmsg = []
    for i in data[column]:
        
        
        
        i = i.replace("hasn't","has not").replace("weren't","were not").replace("don't","do not").replace("mustn't","must not").replace("mightn't","might not").replace("shouldn't","should not").replace("wasn't","was not").replace("needn't","need not").replace("aren't","are not").replace("couldn't","could not").replace("isn't","is not").replace("didn't","did not").replace("won't","would not").replace("wouldn't","would not").replace("shan't","shall not").replace("doesn't","does not").replace("hadn't","had not").replace("haven't","have not") 
        i = i.replace('American Airlines',' AAL ')
        i = i.replace('Amazon', ' AMZN ')
        i = i.replace('Advanced Micro Devices',' AMD ')
        i = i.replace('Align Technology',' ALGN ')
        i = i.replace('Analog Devices',' ADI ')
        i = i.replace('Apple',' AAPL ')
        i = i.replace('Applied Materials',' AMAT ')
        i = i.replace('Automatic Data Processing',' ADP ')
        i = i.replace('Autodesk',' ADSK ')
        i = i.replace('Broadcom',' AVGO ')
        i = i.replace('Baidu',' BIDU ')
        i = i.replace('Biogen',' BIIB ')
        i = i.replace('Biomarin Pharmaceutical',' BMRN ')
        i = i.replace('Cadence Design Systems',' CDNS ')
        i = i.replace('Celgene',' CELG ')
        i = i.replace('Cerner',' CERN ')
        i = i.replace('Ctrip.Com','CTRP')
        i = i.replace('Check Point Software ',' CHKP ')
        i = i.replace('Charter Communications',' CHTR ')
        i = i.replace('Dollar Tree',' DLTR ')
        i = i.replace('Electronic Arts',' EA ')
        i = i.replace('Alphabet',' GOOGL ')
        i = i.replace('Alphabet',' GOOGL ')
        i = i.replace('Google',' GOOGL ')
        i = i.replace('Henry Schein',' HSIC ')
        i = i.replace('Illumina Inc',' ILMN ')
        i = i.replace('Incyte',' INCY ')
        i = i.replace('Intel',' INTC ')
        i = i.replace('Intuit',' INTU ')
        i = i.replace('Intuitive Surgical',' ISRG ')
        i = i.replace('IDEXX',' IDXX ')
        i = i.replace('J.B. Hunt',' JBHT ')
        i = i.replace('JD.com',' JD ')
        i = i.replace('KLA-Tencor',' KLAC ')
        i = i.replace('Kraft Heinz',' KHC ')
        i = i.replace('Lam Research',' LRCX ')
        i = i.replace('Liberty Global',' LBTYA ')
        i = i.replace('Lululemon Athletica',' LULU ')
        i = i.replace('MercadoLibre',' MELI ')
        i = i.replace('Marriott International',' MAR ')
        i = i.replace('Microchip Technology',' MCHP ')
        i = i.replace('Mondelez International',' MDLZ ')
        i = i.replace('Monster Beverage',' MNST ')
        i = i.replace('Microsoft',' MSFT ')
        i = i.replace('Micron Technology',' MU ')
        i = i.replace('Maxim Integrated',' MXIM ')
        i = i.replace('Mylan',' MYL ')
        i = i.replace('NetApp',' NTAP ')
        i = i.replace('Netflix',' NFLX ')
        i = i.replace('NetEase',' NTES ')
        i = i.replace('NVIDIA',' NVDA ').replace('Nvidia','NVDA')
        i = i.replace('NXP Semiconductors',' NXPI ')
        i = i.replace('Paychex',' PAYX ')
        i = i.replace('PACCAR ',' PCAR ')
        i = i.replace('Booking Holdings',' BKNG ')
        i = i.replace('PayPal Holdings',' PYPL ')
        i = i.replace('PepsiCo',' PEP ')
        i = i.replace('Qualcomm',' QCOM ')
        i = i.replace('Regeneron Pharmaceuticals',' REGN ')
        i = i.replace('Ross Stores',' ROST ')
        i = i.replace('Sirius XM',' SIRI ')
        i = i.replace('Skyworks Solutions',' SWKS ')
        i = i.replace('Starbucks',' SBUX ')
        i = i.replace('Symantec',' SYMC ')
        i = i.replace('Synopsys',' SNPS ')
        i = i.replace('Take-Two Interactive',' TTWO ')
        i = i.replace('Tesla',' TSLA ')
        i = i.replace('Texas Instruments',' TXN ')
        i = i.replace('T-Mobile US',' TMUS ')
        i = i.replace('Ulta Beauty',' ULTA ')
        i = i.replace('United Continental Holdings',' UAL ')
        i = i.replace('Verisign',' VRSN ')
        i = i.replace('Verisk Analytics',' VRSK ')
        i = i.replace('Vertex Pharmaceuticals',' VRTX ')
        i = i.replace('Walgreens Boots Alliance',' WBA ')
        i = i.replace('Western Digital',' WDC ')
        i = i.replace('Workday Inc',' WDAY ')
        i = i.replace('Willis Towers Watson',' WLTW ')
        i = i.replace('Wynn Resorts',' WYNN ')
        i = i.replace('Xcel Energy',' XEL ')
        i = i.replace('Xilinx',' XLNX ')
        i = i.replace('Incs','')
        i = i.replace(' Inc ','').replace(' inc ','').replace(' INC ','').replace('Incs','')
        
        i = i.replace('flattolow','flat to low')
        '''i = i.replace(' u ','')
        i = i.replace(' p ','')
        i = re.sub(r'\d+', ' ', i)
        i = i.replace(',','COMMA SAPTARSHI')
        i = i.replace('.','FULL STOP SAPTARSHI')
        #i = i.translate(str.maketrans(' ',' ',string.punctuation))
        i = i.replace('COMMA SAPTARSHI',',')
        i = i.replace('FULL STOP SAPTARSHI','.')
        #i = i.replace('U.S','US')
        #i = i.replace(pat,'')
        i = i.replace('‘','')
        i = i.replace('’','')
        i = i.replace('—',' ')'''
        i = i.replace('\\n',' ')
        #i = i.replace('#','').replace('$','').replace('%','').replace('&','').replace('*','').replace('+','').replace('/','')
        i = i.strip()
        

    
         
       
        result = [j for j in i.split(' ') if len(j)>1]
        
        kmsg.append(' '.join(result))
       
       
    
    data[column] = kmsg
    return data

"Filter the news which contains any of the word in fin_word and creates a single file which contains the financial news"
def find_all_fin_news(frame,fin_word):
    title_im = frame['title']
    date_im = frame['published']
    msg_im = frame['messages']
    full_im = frame['full-text']
    news_set = {}
    fin_title = []
    fin_message = []
    fin_full = []
    date_time = []
    for i in range(len(full_im)):
        for j in fin_word:
            if j.lower() in full_im[i].lower():
                fin_title.append(title_im[i])
                fin_message.append(msg_im[i])
                fin_full.append(full_im[i])
                date_time.append(date_im[i])
                break
    news_set['date'] = date_time
    news_set['title'] = fin_title
    news_set['message'] = fin_message
    news_set['full-text'] = fin_full

"Filter news for each key in the fin_word list and makes separate file for each key word"    
def find_key(frame,fin_word):
    title = frame['title']
    time = frame['publication']
    msg = frame['messages']
    full = frame['full-text']
    statistics = {}
    for j in fin_word:  
        fin_news = []
        label = []
        fin_msg = []
        fin_full = []
        count = 0
        c = 0
        new_dict = {}
        for i in range(len(msg)):
            if j.lower() in msg[i].lower():
                 #print(msg[i])
                 c = c+1
                 count = count+1
                 fin_news.append(title[i])
                 label.append(time[i])
                 fin_msg.append(msg[i])
                 fin_full.append(full[i])
        statistics[j] = count
        new_dict['title'] = fin_news
        new_dict['time'] = label
        new_dict['messages'] = fin_msg
        new_dict['full-text'] = fin_full
        frame = pd.DataFrame.from_dict(new_dict)
        csv = frame.to_csv(j+'.csv',index=False)
        print(csv)
    return statistics   
#print(find_key(kmsg,fin_word))

path_news = r'C:\\Users\\Acer\\Documents\\Python Scripts\\stock_articles_news_set_full_text.csv'
data = pd.read_csv(path_news)
data = data.dropna()
data = data.drop_duplicates()
data =  data_cleaning(data)
"Provide the method whichever you want to use"
print(find_all_fin_news(data,news_fin_list))