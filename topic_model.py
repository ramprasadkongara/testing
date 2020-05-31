import os
os.chdir('/opt/app-root/src/')
import re
import glob
import gensim
import pyLDAvis
import pyLDAvis.gensim
import spacy
import pandas as pd
import nltk; nltk.download('stopwords')
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import re
import warnings
from pprint import pprint
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
#%config InlineBackend.figure_formats = ['retina']
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt
from collections import OrderedDict
import scipy
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import glob
import seaborn
import re
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
from nltk.corpus import stopwords
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
sns.set(color_codes=True)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize
from nltk.util import ngrams
import gensim
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis

files = glob.glob('Data/incident*.xlsx')
raw = pd.concat([pd.read_excel(f, sheet_name='Page 1') for f in files],ignore_index=True)
df = raw 
df = df.drop(['Parent','Customer Impact','Parent Incident','Assignment group','Assigned to','Actions taken','Activity due','Actual start','Additional assignee list','Additional comments','Approval','Approval history','Approval set','Business Contact','Business impact','Cause CI','Caused by Change','Closed by','Created by','Geography','KB Article to be created?','Location','Opened by','SLA due','Updated','Updated by','Active','Closed'], axis=1)

df["Caller"] = df["Caller"].str.strip()
DataNoevent = df[df['Caller'] != "Event Management"]
df = DataNoevent

df_app =pd.read_csv('/opt/app-root/src/ApplicationNames.csv',encoding='ISO-8859-1')
df['Short_long'] = df['Short description'].map(str) + df['Description'].map(str)
df['Short_long'] = df['Short_long'].str.replace("nan", "")
#df['Short_long']

def ApplicationNameCI(s):
    for x, y in zip(df_app['Application Name'], df_app['Short Name']):
           if x in s:
                return y

df['Application'] = df['Configuration item'].apply(lambda x: ApplicationNameCI(x))

df['Application'] = df['Application'].map(str)
df['Business service'] = df['Business service'].map(str)

app=[]
for index, row in df.iterrows(): 
    if (row['Application']=='None'):
        app =ApplicationNameCI(row['Business service'])
        df['Application'][index]=app 

def getappname(des):
    appname=[]
    application_name = ['ASW2','CFMS','CLW','HLP','SBOS','RLM','BOE','PRS','CSW','UAM','APEA','ASW','Cers','EDW','BIH','WAS',
     'WOC','PRPC','Disputes','DCS']
#     application_name = ['asw2','cfms','clw','hlp','sbos','rlm','boe','prs','csw','uam',
#                     'apea','asw','cers','edw','bih','woc','prpc','disputes','dcs']  
    s1 = set(des.split())
    s2=set(application_name)
    appname=(s1.intersection(s2))
    if appname:
        return ', '.join(appname)
    else:
        return''

df['App']=[getappname(r) for r in df['Short_long']]

def checkappnames(df):
    App1=df[0]
    App2=df[1]
    if App1 and App2:
        if App1!=App2:
            return App2
        else:
            return App2
    elif App1:
        return App1
    elif App2:
        return App2
    else:
        return ""

df['NewCol'] = df[['Application', 'App']].apply(checkappnames, axis=1)

def split(df, column, sep='|', keep=False):
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df

df = split(df, 'NewCol', sep=',')
df = df.filter(['Number','Opened','Short_long', 'NewCol'], axis = 1)
df= df.rename(columns = {'NewCol':'AppName'})

#remove duplicates
list_values = []
for i in df['Short_long']:
    list_values.append(i)

removed_duplicate = []
for i in list_values:
    #s = i
    cleared = ' '.join(OrderedDict((w,w) for w in i.split()).keys())
    removed_duplicate.append(cleared)
df['without_duplicate'] = removed_duplicate

df.dropna(inplace=True)

df_regex = df[['without_duplicate']].copy()

def regex00(df):
    short = []
    duplicate = []
    for index, rows in df.iterrows():
        a = rows[0]
        replace_text = ''
        reg='STEPS TAKEN TO RESOLVE INCIDENT.*?tips.aspx|STEPS TAKEN TO RESOLVE INCIDENT.*?\w{2}\S\d{4}|From:.*?Subject:|The only workaround.*?business mentioned above|Connect Issue.*?#techend|Preferred Contact No.*?#Techend|ICBOX.*?#techend|Will EU.*?Alternate|Number.*?Current Address|Contact Number.*?name number|Will EU.*?name number|Number users affected.*?AU 3008|Outage.*?contact numbers|Number.*?BANGALORE KA IN|Preferred Contact No.*?VDI connection type|Number users impacted.*?AU 3008|Number of users impacted.*?PH 1110|E.g. Mailbox Name.*?Docklands VIC 3008|Number of users affected.*?Auckland 1010|Number of users affected.*?Manyata Embassy Business Park- SEZ, BANGALORE KA IN|Number of users impacted.*?Manyata|Number of users impacted.*?BANGALORE  KA|Number of users affected.*?KA|Number of users impacted.*?NZ  1010|Number of users impacted.*?AU  /d{4}|Thanks and regards.*?V\w{2} 3008|Report.*?tips.aspx|Number of users impacted.*?AU  3000|If issue is not resolved.*?when escalating|Number of users impacted.*?Bangalore, India|Number of users impacted.*?Address|Number of users impacted.*?CN  610041|Number of users impacted.*?AU  3121|Number of users impacted.*?AU 3008|Number of users impacted.*?PH  1110|'
        reg+= 'Data cap.*?flowing|Splunk.*?Circum|Inbound path.*?aup4330l|Path.*?AUP4330L|path.*?PEGA_BIX|Path.*?PEGA_BIX|Preferred Contact.*?Affected|UpperGround.*?3008||When.*?recently|ANZ Support Services.*?BANGALORE KA IN'
        regex = re.compile(reg,re.DOTALL)
        pattern = re.findall(regex, a)
        pattern = list(set(pattern))
        pattern = ','.join(pattern)
        out = re.sub(regex, replace_text, a)
        short.append(out)
        duplicate.append(pattern)
    df['struc_short'] = short
    df['Features'] = duplicate
    df.to_csv('/opt/app-root/src/Regexoutput/regex00.csv')
    
def regex01(df):    
    short = []
    duplicate = []
    for index, rows in df.iterrows():
        a = rows[0]
        replace_text = ' '
        reg ='PRB\d{7}|ANZ[a-zA-Z0-9]{6,}-\d{1,}|CNWS-\d{8}|\W{3,}URGENT\W{2,}|\W{3}URGENT ASSISTANCE\W{3}|\w{4}\s\w{5}\s\w{3}:\s\w-\d{7}|SR-\d{7}|ANZ-AUS-\w{3}\W\w{4}|\W{3}no auto ticket\W{3}|ANZ-SW-Work I-\d{5,}|I-\d{5,}|RLMAPP-\d{6}|S-\d{5,}|CapCisID: \d{6,}|Lodgementid:\d{8,}|\W{1,}no auto ticket\W{1,}|RLMINDX-\d{4,}|TRF-\d{2,}|\w{3,}\syou\sin\w{4,}'
        regex = re.compile(reg)
        pattern = re.findall(regex, a)
        pattern = list(set(pattern))
        pattern = ','.join(pattern)
        out = re.sub(regex, replace_text, a)
        short.append(out)
        duplicate.append(pattern)
    df['struc_short'] = short
    df['Features'] = duplicate
    df.to_csv('/opt/app-root/src/Regexoutput/regex01.csv')
    
def regex02(df):    
    short = []
    duplicate = []
    for index, rows in df.iterrows():
        a = rows[0]
        replace_text = ' '
        reg='App\s\w{2}:\s\d{9}|Sr-\d{7}|Salary ID|Application\s\w{5}\s\w{2}:|\(\w{6}\W\w{7}\W\w{4}\s\w{6}\)|\(\w{6}\W\w{7}\W\w{4}\s\w{6}\s\w{3}\s\w{7}\)|\(laptop/desktop/VDI/Mobile Device Type\)|\w{5}\d{6}\-\d{5}|\(\w{2}\s\w{5}\s\w{3}\)|\w{5}\s\w{8}\s\(\w{1}\/\w{1}\)|Application\s\d{10}|Issue\s\w{11}\:|Reported\s\w{5}\s\W|\*no\s\w{4}\s\w{6}\*|is\w{4}\:|mcp\w{4}\d{1}|JI\w{2}\s\d{3}|SDM\s\d{8}|no\s\w{4} incident|no\s\w{4} incidents|aup\d{4}\w{1}|LAN \w{2}\W |Error \w{4}age|Screen\w{4} \w{4}ched|Screen\w{4} \w{5}tory|Reference no'
        regex = re.compile(reg)
        pattern = re.findall(regex, a)
        
        pattern = list(set(pattern))
        pattern = ', '.join(pattern)
        out = re.sub(regex, replace_text, a)
        short.append(out)
        duplicate.append(pattern)
    df['struc_short'] = short
    df['Features'] = duplicate
    df.to_csv('/opt/app-root/src/Regexoutput/regex02.csv')
    
def regex03(df):
    short = []
    duplicate = []
    for index, rows in df.iterrows():
        a = rows[0]
        replace_text = ''
        reg = 'DPAMF\d{6}\-\d{5}|TRF-\d{2,}|\w{3,}\syou\sin\w{4,}|RLMINDX-\d{4,}|S-\d{2,}|SR-\d{2,}|RLMAPP:\s\d{5}|APP ID: RLMAPP-/d{5}|LTD\d{6}\w{6}\d{4}|LTD\s\d{6}\w{6}|AO-|TSM - AO PEGA Support|ANZSDIAUSHistory\w{5,}|'
        reg+='Pega Id\s\-\s\w{7}|LAN ID:\s\w{3,8}|APP\s\d{10}|app\s\d{10}|App\s\d{10}|APP\s#\s\d{10}|ANZRLM_CAW_HISTORY|Example :\s\d{8}|support group|Screenshot|Desktop Director'
        regex = re.compile(reg)
        pattern = re.findall(regex, a)
        pattern = list(set(pattern))
        pattern = ','.join(pattern)
        out = re.sub(regex, replace_text, a)
        short.append(out)
        duplicate.append(pattern)
    df['struc_short'] = short
    df['Features'] = duplicate
    df.to_csv('/opt/app-root/src/Regexoutput/regex03.csv')
    
def regex04(df):
    short = []
    duplicate = []
    for index, rows in df.iterrows():
        a = rows[0]
        replace_text =''
        reg = 'Workstation ID : \w{4}-\d{8}|Workstation ID : \w{4}-[a-zA-Z0-9]{7,}|workstation id : \w{4}-[a-zA-Z0-9]{7,}|Workstation : [a-zA-Z0-9]{1,}|Workstation ID : [^a-z]{14}|Workstation ID  : [^a-z]{13}|Workstation ID : [^a-z]{13}|Workstation ID : [^a-z]{12}|WORKSTATION\s[a-zA-Z0-9]{1,}|'
        reg+='workstation ID: \w{12}|Workstation ID: \w{2}-D\w{12}|\sWorkstation ID : \d{8,}|Workstation ID : d{8,}|Workstation ID : NA|Workstation ID : L\w{11,}|Workstation ID : VP\w{12}|Workstation ID  : l\w{12}|Workstation ID  :   \w{4}-\w{8}|Workstation ID  :VP\w{12}|Workstation ID  : d\w{12}|Workstation ID: d\w{12}|Workstation ID  : D\w{12}|Workstation ID  : inlt-\w{8}'
        regex = re.compile(reg)
        pattern = re.findall(regex, a)
        pattern = list(set(pattern))
        pattern = ','.join(pattern)
        out = re.sub(regex, replace_text, a)
        short.append(out)
        duplicate.append(pattern)
    df['struc_short'] = short
    df['Features'] = duplicate
    df.to_csv('/opt/app-root/src/Regexoutput/regex04.csv')
    
def regex05(df):
    short = []
    duplicate = []
    for index, rows in df.iterrows():
        a = rows[0]
        replace_text =''
        reg = 'workstation id: L\w{11,}|workstation id: D\w{11,}||\WWorkstation ID : \w{1}\/\w{1}|\WWorkstation ID :|Workstation ID  :'
        regex = re.compile(reg)
        pattern = re.findall(regex, a)
        pattern = list(set(pattern))
        pattern = ','.join(pattern)
        out = re.sub(regex, replace_text, a)
        short.append(out)
        duplicate.append(pattern)
    df['struc_short'] = short
    df['Features'] = duplicate
    df.to_csv('/opt/app-root/src/Regexoutput/regex05.csv')
    
def regex06(df):
    short = []
    duplicate = []
    for index, rows in df.iterrows():
        a = rows[0]
        replace_text =''
        reg = '\d{1,}\/\d{2}\/\d{2,}|\d{1,}\-\d{2}\-\d{2,}|\d{2}\:\d{2}\:\d{2} [ap]m|\d{2}\:\d{2} [ap]m|\d{4}\W\d{2}\W\d{2}|\d{1,}\w{2}\s\w{3,}\s\d{1,}'
        regex = re.compile(reg)
        pattern = re.findall(regex, a)
        pattern = list(set(pattern))
        pattern = ','.join(pattern)
        out = re.sub(regex, replace_text, a)
        short.append(out)
        duplicate.append(pattern)
    df['struc_short'] = short
    df['Features'] = duplicate
    df.to_csv('/opt/app-root/src/Regexoutput/regex06.csv')
    
def regex07(df):
    short = []
    duplicate = []
    for index, rows in df.iterrows():
        a = rows[0]
        replace_text = ''
        reg='Ã‚|Ãƒ|Â¢|\n|\x82|\x83|\x80'
        #'Ã‚|Ãƒ|'
        #\x82Ãƒ\x82Ã‚\x95|Ãƒâ€šÃ‚â€šÃƒÆ’Ã‚â€šÃƒâ€šÃ‚|Ã‚\x82Ãƒ\x82Ã‚\x94|Ã‚\x82Ãƒ\x82Ã‚\x93|Ã‚\x82Ãƒ\x82Ã‚\x92afaÃ‚\x82Ãƒ\x82Ã‚\x92|Ã‚\x82Ãƒ\x82Ã‚\x92|Ã‚\x82Ãƒ\x82Ã‚\x91|Ã‚\x82Ãƒ\x82Ã‚\x96|Ã‚\x82Ãƒ\x82Ã‚|Ã‚\x82Ãƒ\x82Ã‚Â®|Ãƒ\x82Ã‚\x94|Ã‚\x82Ãƒ\x82Ã‚|'
        regex = re.compile(reg)
        pattern = re.findall(regex, a)
        pattern = list(set(pattern))
        pattern = ','.join(pattern)
        out = re.sub(regex, replace_text, a)
        short.append(out)
        duplicate.append(pattern)
    df['struc_short'] = short
    df['Features'] = duplicate
    df.to_csv('/opt/app-root/src/Regexoutput/regex07.csv')
    
def regex08(df):
    short = []
    duplicate = []
    for index, rows in df.iterrows():
        a = rows[0]
        replace_text = ''
        reg='_|sr-\d{6,}|System Issue Description|Please|please|laptop/desktop/VDI/Mobile Device Type|Original Message|Contacts(Name and number)|Pre Post TVT Query|Salary ID|Lan ID|User best form contact|Screenshot of|Desktop Director|pegajupiter_data.anz_anzsdiaus_work_dispute|pegajupiter_data.ANZ_ANZSDCAUS_DATA_SCRPTENT|Message|Screenshot Mandatory|ANZ[a-zA-Z0-9]{6,}-\d{1,}|Visa OnUs \d+\s\d+|Visa OnUs Fraud \d+\s\d+|Visa OnUs Fraud \d+\s\s\d+|\S+@\S+|User best \w{4}\s\w{2}\s\w{7}\:|Access/Application/Profile/system|Application/system|Printer/EV/Right fax|laptop/desktop/thin client|BuildingManyata|Ltd.,Eucalyptus|MAC Address|E.g. Mailbox Name/ Server details|'
        reg+='Contacts\(Name number\)|Contacts\(Name and number\)|path\s\W{2}10.44.130.\d{1,}\Wpegamis\WDownload|833 Collins St, Docklan VIC 3008|Workstation|Kindly|screenshot\/attached|screenshot attached|screenshots attached|RLM ID :|RLMAPP-\d{6}|\w{3,}.\w{2,}@anz.com|\W\w\W\w\W\W\s\w{1,}|STEPS TAKEN TO RESOLVE INC\w{1,}:|System Issue Description:|Logon ID: \w{2,}|\w{3,}\s\w{7}\W\w{7}\s\w{7,}\W\s\w{4}|\w{10,}\s\w{2}\s\W\w{6}\W\w{7}\W\w{3}\W\w{5,}\s\w{3,}\s\w{4}\W|\w{10,}\s\w{2}\s|'
        reg+='ERR-:\s\d{10}|ANZBAU\d{1}\w{5}\d{23}|Lodgementid:\d{8,}|\d{1,}\W|I-\d{1,}|path\s\W{2}10.44.130.\d{1,}\Wpegamis\WDownload|\d{2}\W\d{2}\W\d{4}|nan|RLM ID :|RLMAPP-\d{6}|\w{3,}.\w{2,}@anz.com|\W\w\W\w\W\W\s\w{1,}'
        regex = re.compile(reg)
        pattern = re.findall(regex, a)
        pattern = list(set(pattern))
        pattern = ','.join(pattern)
        out = re.sub(regex, replace_text, a)
        short.append(out)
        duplicate.append(pattern)
    df['struc_short'] = short
    df['Features'] = duplicate
    df.to_csv('/opt/app-root/src/Regexoutput/regex08.csv')
    
def regex09(df):
    short = []
    duplicate = []
    for index, rows in df.iterrows():
        a = rows[0]
        replace_text = ''
        reg='\w{4}PPRGMER|\w{4}Hi|\w{1}eam\w{2}|\w{1}\d{2}ggp|Files:Offcr\w{50,}Chnl|\w{1}egards|\w{1}uilding|\w{1}anyata|'
        reg+='\w{1}ucalyptus|\w{1}mbassy|\w{1}usiness|\w{1}ark|SEZ|KAIN|BANGALORE|BG S|S\d{2}|ANZ\w{3}AUS|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*|'
        reg+='PEGAUPMIS\w{3,}|0uzv\w{1,}|0u\w{3}|HISTORYANZFWS\w{10,}'
        regex = re.compile(reg)
        pattern = re.findall(regex, a)
        pattern = list(set(pattern))
        pattern = ','.join(pattern)
        out = re.sub(regex, replace_text, a)
        short.append(out)
        duplicate.append(pattern)
    df['struc_short'] = short
    df['Features'] = duplicate
    df.to_csv('/opt/app-root/src/Regexoutput/regex09.csv')

regex00(df_regex)
df_01 = pd.read_csv('/opt/app-root/src/Regexoutput/regex00.csv',encoding='ISO-8859-1')
df_01.drop(['without_duplicate','Features'],axis = 1, inplace = True)
df_01 = df_01['struc_short'].astype(str)
df_01 = df_01.to_frame()
df_01 = df_01.rename(columns = {'struc_short':'Short_long'})

regex01(df_01)
df_02 = pd.read_csv('/opt/app-root/src/Regexoutput/regex01.csv',encoding='ISO-8859-1')
df_02.drop(['Short_long','Features'],axis = 1, inplace = True)
df_02 = df_02['struc_short'].astype(str)
df_02=df_02.to_frame()
df_02=df_02.rename(columns = {'struc_short':'Short_long'})

regex02(df_02)
df_03 = pd.read_csv('/opt/app-root/src/Regexoutput/regex02.csv',encoding='ISO-8859-1')
df_03.drop(['Short_long','Features'],axis = 1, inplace = True)
df_03 = df_03['struc_short'].astype(str)
df_03 = df_03.to_frame()
df_03 = df_03.rename(columns = {'struc_short':'Short_long'})

regex03(df_03)
df_04 = pd.read_csv('/opt/app-root/src/Regexoutput/regex03.csv',encoding='ISO-8859-1')
df_04.drop(['Short_long','Features'],axis = 1, inplace = True)
df_04 = df_04['struc_short'].astype(str)
df_04 = df_04.to_frame()
df_04 = df_04.rename(columns = {'struc_short':'Short_long'})

regex04(df_04)
df_05 = pd.read_csv('/opt/app-root/src/Regexoutput/regex04.csv',encoding='ISO-8859-1')
df_05.drop(['Short_long','Features'],axis = 1, inplace = True)
df_05 = df_05['struc_short'].astype(str)
df_05 = df_05.to_frame()
df_05 = df_05.rename(columns = {'struc_short':'Short_long'})

regex05(df_05)
df_06 = pd.read_csv('/opt/app-root/src/Regexoutput/regex05.csv',encoding='ISO-8859-1')
df_06.drop(['Short_long','Features'],axis = 1, inplace = True)
df_06 = df_06['struc_short'].astype(str)
df_06 = df_06.to_frame()
df_06 = df_06.rename(columns = {'struc_short':'Short_long'})

regex06(df_06)
df_07 = pd.read_csv('/opt/app-root/src/Regexoutput/regex06.csv',encoding='ISO-8859-1')
df_07.drop(['Short_long','Features'],axis = 1, inplace = True)
df_07 = df_07['struc_short'].astype(str)
df_07 = df_07.to_frame()
df_07 = df_07.rename(columns = {'struc_short':'Short_long'})

regex07(df_07)
df_08 = pd.read_csv('/opt/app-root/src/Regexoutput/regex07.csv',encoding='ISO-8859-1')
df_08.drop(['Short_long','Features'],axis = 1, inplace = True)
df_08 = df_08['struc_short'].astype(str)
df_08 = df_08.to_frame()
df_08 = df_08.rename(columns = {'struc_short':'Short_long'})

regex08(df_08)
df_09 = pd.read_csv('/opt/app-root/src/Regexoutput/regex08.csv',encoding='ISO-8859-1')
df_09.drop(['Short_long','Features'],axis = 1, inplace = True)
df_09 = df_09['struc_short'].astype(str)
df_09 = df_09.to_frame()
df_09 = df_09.rename(columns = {'struc_short':'Short_long'})

regex09(df_09)
df_10 = pd.read_csv('/opt/app-root/src/Regexoutput/regex09.csv',encoding='ISO-8859-1')
df_10.drop(['Short_long','Features'],axis = 1, inplace = True)
df_10 = df_10['struc_short'].astype(str)
df_10 = df_10.to_frame()
df_10 = df_10.rename(columns = {'struc_short':'Short_long'})

df_10['Short_long'] = df_10['Short_long'].str.lower()
df_10['Short_long'] = df_10['Short_long'].str.replace('\W',' ')

df['struc_short']=df_10['Short_long'].values
df.dropna(inplace=True)

stop_words = set(stopwords.words("english"))
add_stopwords = open("/opt/app-root/src/Stopwords.txt", mode="r", encoding='ISO-8859-1').read().split()
stop_words = stop_words.union(add_stopwords)

def stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words ])
    return rev_new



df['struc_short'] = [stopwords(r.split()) for r in df['struc_short']]
df['struc_short'] = df['struc_short'].str.replace('\d+','')

df['single_token'] = df.apply(lambda row:nltk.word_tokenize(row['struc_short']), axis=1)

ConvBigrams=[]
for doc in df['single_token']:
        
    #text=doc.split()
    bigrams_list = list(ngrams(doc,2))
    new_list = []
    for words in range(len(bigrams_list)):
        if(bigrams_list[words][0] != bigrams_list[words][1]):
            new_list.append('_'.join(bigrams_list[words]))
    
    ConvBigrams.append(new_list)
    
df['bigram'] = ConvBigrams
df['bigram'] = ConvBigrams
df['bigram'] = df['bigram'].astype(str)

def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0] 
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(df['bigram'])
        
# Tweak the two parameters below
number_topics = 4
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
#print("Topics found via LDA:")
#print_topics(lda, count_vectorizer, number_words)

single_topic = lda.components_[0]
single_topic.argsort()
single_topic[100]
single_topic[427]
single_topic.argsort()[-10:]
top_word_indices = single_topic.argsort()[-10:]
topic_results = lda.transform(count_data)
topic_results[0].round(2)
topic_results[0].argmax()
topic_results.argmax(axis=1)
df['Topic'] = topic_results.argmax(axis=1)
df.Topic = df.Topic.map(str)

df['Topic'] = df['Topic'].replace({'0': 'Access_Connection_issues','1': 'Incident_status_issues','2': 'Files_document_issues','3': 'Service_request_issues'})

print(len(df[df['Topic'] == "Access_Connection_issues"]))
print(len(df[df['Topic'] == "Incident_status_issues"]))
print(len(df[df['Topic'] == "Files_document_issues"]))
print(len(df[df['Topic'] == "Service_request_issues"]))
df.to_csv('/opt/app-root/src/lda_gensim.csv')

#%%time
LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(number_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)

with open(LDAvis_data_filepath, 'wb') as f:    
    pickle.dump(LDAvis_prepared, f)
        
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(number_topics) +'.html')

