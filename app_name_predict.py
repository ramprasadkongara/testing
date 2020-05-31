import re
import pandas as pd
import os
import glob
import numpy as np
os.chdir('/opt/app-root/src/Data/')

files = glob.glob('incident*.xlsx')
raw = pd.concat([pd.read_excel(f, sheet_name='Page 1') for f in files],ignore_index=True)

df = raw
df = df.drop(['Parent','Customer Impact','Parent Incident','Assignment group','Assigned to','Actions taken','Activity due','Actual start','Additional assignee list','Additional comments','Approval','Approval history','Approval set','Business Contact','Business impact','Cause CI','Caused by Change','Closed by','Created by','Geography','KB Article to be created?','Location','Opened by','SLA due','Updated','Updated by','Active','Closed'], axis=1)

df["Caller"] = df["Caller"].str.strip()
DataNoevent = df[df['Caller'] != "Event Management"]
DataNoevent.shape

DataNoevent['Caller'].unique()
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

df['NewCol'].value_counts()[:20].plot(kind='bar')

def split(df, column, sep='|', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
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

df['NewCol'].value_counts()[:20].plot(kind='bar')

df = df.filter(['Short_long', 'NewCol'], axis = 1)

#remove duplicates
list_values = []
for i in df['Short_long']:
    list_values.append(i)

from collections import OrderedDict
removed_duplicate = []
for i in list_values:
    #s = i
    cleared = ' '.join(OrderedDict((w,w) for w in i.split()).keys())
    removed_duplicate.append(cleared)
df['without_duplicate'] = removed_duplicate

#df.isnull().sum()
df.dropna(inplace=True)

#df['NewCol'].value_counts()
filter_df_high_app = df["NewCol"].isin(["ASW", "ASW2", "RLM", "PRS", "SBOS", "Disputes", "Pega - Archive and Purge", "CERS", "BOE"]) 

df = df[filter_df_high_app]

from sklearn.model_selection import train_test_split
X = df['without_duplicate']
y = df['NewCol']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Linear SVC:
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC())])

text_clf_lsvc.fit(X_train, y_train)

# Form a prediction set
predictions = text_clf_lsvc.predict(X_test)

from sklearn import metrics
#print(metrics.confusion_matrix(y_test,predictions))

#print(metrics.classification_report(y_test,predictions))

#print(metrics.accuracy_score(y_test,predictions))

description = "ANZAU200120-00465, kindly reject the case as moved to GAMSActionWB  while rejecting the case"

print(text_clf_lsvc.predict([description]))