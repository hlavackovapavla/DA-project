from oauth2client.client import OAuth2WebServerFlow
from oauth2client.tools import run_flow
from oauth2client.file import Storage
import json
import os
import re
import httplib2 
from oauth2client import GOOGLE_REVOKE_URI, GOOGLE_TOKEN_URI, client
import requests
import pandas as pd
import datetime
import time
import pyodbc
import sqlalchemy as sal
from sqlalchemy import create_engine
import csv
import numpy as np
import collections
from itertools import chain
import itertools
from scipy.stats import stats
import statistics
import sys
import yaml

#function for checking whether file exist in the path or not
def jsonExist(file_name):
  return os.path.exists(file_name)

#function that returns the refresh token
def get_refresh_token(client_id,client_secret):
  CLIENT_ID = client_id
  CLIENT_SECRET = client_secret
  SCOPE = 'https://www.googleapis.com/auth/analytics.readonly'
  REDIRECT_URI = 'http://localhost:8080/'
  flow = OAuth2WebServerFlow(client_id=CLIENT_ID,client_secret=CLIENT_SECRET,scope=SCOPE,redirect_uri=REDIRECT_URI)
  if jsonExist('credential.json')==False:
    storage = Storage('credential.json') 
    credentials = run_flow(flow, storage)
    refresh_token=credentials.refresh_token
  elif jsonExist('credential.json')==True:
    with open('credential.json') as json_file:  
      refresh_token=json.load(json_file)['refresh_token']
  return(refresh_token)


#load client_id and client_secret from config
with open("config.yaml", 'r') as stream:
    try:
        client_id = yaml.safe_load(stream)
        client_secret = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

refresh_token=get_refresh_token(client_id,client_secret)

#function return the multi channel data for given dimension, metrics, start data, end data access token, transaction_type, startindex
def google_analytics_multi_channel_funnel_reporting_api_data_extraction(viewID,dim,met,start_date,end_date,refresh_token,transaction_type,startIndex):
  metric="%2C".join([re.sub(":","%3A",i) for i in met])
  dimension="%2C".join([re.sub(":","%3A",i) for i in dim])
  dimension=dimension+"&filters=mcf%3AconversionType%3D%3D"+transaction_type+"&samplingLevel=HIGHER_PRECISION&max-results=100000"
  if jsonExist('credential.json')==True:
    with open('credential.json') as json_file:
      storage_data = json.load(json_file)
    client_id=storage_data['client_id']
    client_secret=storage_data['client_secret']
    credentials = client.OAuth2Credentials(access_token=None, client_id=client_id, client_secret=client_secret, refresh_token=refresh_token, token_expiry=3600,token_uri=GOOGLE_TOKEN_URI,user_agent='my-user-agent/1.0',revoke_uri=GOOGLE_REVOKE_URI)
    credentials.refresh(httplib2.Http())
    rt=(json.loads(credentials.to_json()))['access_token']  
    api_url="https://www.googleapis.com/analytics/v3/data/mcf?access_token="
    url="".join([api_url,rt,'&ids=ga:',viewID,'&start-date=',start_date,'&end-date=',end_date,'&metrics=',metric,'&dimensions=',dimension, '&start-index=', startIndex,])
    try:
      r = requests.get(url)
      try:
        data=pd.DataFrame(list((r.json())['rows']))
        print("data extraction is successfully completed")
        table=pd.DataFrame()
        if data.shape[0]!=0:
          for i in range(0,data.shape[0]):
            data1=pd.DataFrame()
            data1=(data.iloc[i,:]).tolist()
            for k in range(0,len(data1)):
              if 'conversionPathValue' in data1[k]:
                value_list=[]
                for k1 in range(0,len(data1[k]['conversionPathValue'])):
                  value_list.append(data1[k]['conversionPathValue'][k1]['nodeValue'])
                  table.loc[i,k]=('>').join(value_list)
              elif 'primitiveValue' in data1[k]:
                  table.loc[i,k]=data1[k]['primitiveValue']
          if table.shape[0]!=0:   
            table.columns=[re.sub("mcf:","",i) for i in dim+met]
            table['date']=start_date
            return(table) 
      except:
        print(r.json())
    except:
      print(r.json())

viewID='83873242'
dim=['mcf:sourceMediumPath', 'mcf:campaignPath', 'mcf:conversionType', 'mcf:conversionDate']
met=['mcf:totalConversions', 'mcf:totalConversionValue']
start_date='2020-01-01'
end_date= '2020-11-19'#str(datetime.date.today())
transaction_type='Transaction'
refresh_token=refresh_token
startIndex = '1'

#pagination
columnNames = ['sourceMediumPath', 'campaignPath', 'conversionType', 'conversionDate', 'totalConversions', 'totalConversionValue', 'date']
finalData = pd.DataFrame(columns = columnNames)
while True:
  data = google_analytics_multi_channel_funnel_reporting_api_data_extraction(viewID,dim,met,start_date,end_date,refresh_token,transaction_type,startIndex)
  if data is not None:  
    finalData = pd.concat([finalData, data], ignore_index=True)
    startIndex = str(int(startIndex) + 10000)
  if data is None:
    break

#replace / from values
sourceMediumPathreplaceList = finalData['sourceMediumPath'].tolist()
sourceMediumPathreplace = [sourceMedium.replace(' / ', ' : ').replace('/', '-').replace(' : ', '/') for sourceMedium in sourceMediumPathreplaceList]
finalData['sourceMediumPath'] = sourceMediumPathreplace
campaignPathreplaceList = finalData['campaignPath'].tolist()
campaignPathreplace = [campaign.replace('/', '-') for campaign in campaignPathreplaceList]
finalData['campaignPath'] = campaignPathreplace

#merging Source/Medium/Campaign into one column
dataList = finalData.values.tolist()
sourceMediumPath = [d[0].split('>') for d in dataList]
campaignPath = [d[1].split('>') for d in dataList]
pathLength = [len(path) for path in campaignPath]
sourceMediumCampaignPath = []
listItem = []
for j in range (len(sourceMediumPath)):
  for i in range(len(sourceMediumPath[j])):
    item = sourceMediumPath[j][i] + '/' + campaignPath[j][i]
    listItem.append(item)
  sourceMediumCampaignPath.append(listItem)
  listItem = []
sourceMediumCampaignPath = ['>'.join(item) for item in sourceMediumCampaignPath]
finalData['sourceMediumCampaignPath'] = sourceMediumCampaignPath

#new column with path lenght
finalData['pathLength'] = pathLength

#calculation of the total number of conversions and their values
conversionTotal = sum([float(channel[4]) for channel in dataList])
conversionValueTotal = sum([float(channel[5]) for channel in dataList])

'''
#) napojení se na databázi + nahrání dat do databáze pomocí SQLAlchemy
engine = sal.create_engine('mssql+pyodbc://localhost\\SQLEXPRESS/attributionmodels?driver=SQL Server?Trusted_Connection=yes')
conn = engine.connect()
data.to_sql('attributionModels', con=engine, if_exists='replace',index=True,chunksize=1000)
'''
'''
#) načtení dat z databáze pomocí SQLAlchemy do DF

sql_query = pd.read_sql_query('SELECT * FROM attributionmodels.dbo.attributionModels', engine)
conn.close()
print(sql_query)
'''

#uploading data into database
conn = pyodbc.connect(
  'Driver={SQL Server};'
  'Server=localhost\\SQLEXPRESS;'
  'Database=attributionmodels;'
  'Trusted_Connection=yes;'
  'User=test;'
  'Password=test;'
)
cursor = conn.cursor()
for index,row in finalData.iterrows():
  cursor.execute("INSERT INTO dbo.Atrtributionmodels([sourceMediumPath], [campaignPath], [conversionType], [conversionDate], [totalConversions], [totalConversionValue], [date], [SourceMediumCampaignPath], [PathLength]) values (?, ?, ?, ?, ?, ?, ?, ?, ?)", row['sourceMediumPath'], row['campaignPath'], row['conversionType'], row['conversionDate'], row['totalConversions'], row['totalConversionValue'], row['date'], row['sourceMediumCampaignPath'], row['pathLength']) 
  conn.commit()

#calculation of the Last click model 
result = {}
cursor.execute('SELECT * FROM dbo.Atrtributionmodels')
for row in cursor:
  sourceMediumCampaign = str(row[-2])
  conversionPath = sourceMediumCampaign.split('>') 
  lastSource = conversionPath[-1]
  key = lastSource + ' ; lastclick'
  if key not in result:
    values = lastSource.split('/')
    result[key] = {
    'Source':values[0],
    'Medium':values[1],
    'Campaign':values[2],
    'totalConversions':float(row[4]) ,
    'totalValue':float(row[5]),
    'Model':'lastClick'
    } 
  else:
    result[key]['totalConversions'] += float(row[4]) 
    result[key]['totalValue'] += float(row[5])

#calculation of the First click model
cursor.execute('SELECT * FROM dbo.Atrtributionmodels')
for row in cursor :
  sourceMediumCampaign = str(row[-2])
  conversionPath = sourceMediumCampaign.split('>') 
  firstSource = conversionPath[0]
  key = firstSource + ' ; firstclick'
  if key not in result:
    values = firstSource.split('/')
    result[key] = {
    'Source' : values[0],
    'Medium' : values[1],
    'Campaign' : values[2],
    'totalConversions' :float(row[4]) ,
    'totalValue' : float(row[5]),
    'Model' : 'firstClick'
    } 
  else:
    result[key]['totalConversions'] += float(row[4]) 
    result[key]['totalValue'] += float(row[5])

#calculation of the Linear model
cursor.execute('SELECT * FROM dbo.Atrtributionmodels')
for row in cursor :
  sourceMediumCampaign = str(row[-2])
  conversionPath = sourceMediumCampaign.split('>') 
  for conversion in conversionPath:
    key = conversion + '; linear'
    if key not in result:
      values = conversion.split('/')
      result[key] = {
      'Source' : values[0],
      'Medium' : values[1],
      'Campaign' : values[2],
      'totalConversions' : float(row[4])/len(conversionPath),
      'totalValue' : float(row[5])/len(conversionPath),
      'Model' : 'linear'
      } 
    else:
      result[key]['totalConversions'] += float(row[4])/len(conversionPath)
      result[key]['totalValue'] += float(row[5])/len(conversionPath)
cursor.close()
conn.close()

#following function is used to return the unique list
def unique(list1):  
  unique_list = []   
  for x in list1: 
    if x not in unique_list: 
      unique_list.append(x) 
  return(unique_list)

#following function is used to split the string by '>'
def split_fun(path):
  return path.split('>')

#following function is used to return ranked vector ascending order 
def calculate_rank(vector):
  a={}
  rank=0
  for num in sorted(vector):
    if num not in a:
      a[num]=rank
      rank=rank+1
  return[a[i] for i in vector]

#following function is used to return transition matrix
def transition_matrix_func(import_data):
  z_import_data=import_data.copy()
  z_import_data['path1']='start>'+z_import_data['path']
  z_import_data['path2']=z_import_data['path1']+'>convert'   
  z_import_data['pair']=z_import_data['path2'].apply(split_fun)
  zlist=z_import_data['pair'].tolist()
  zlist=list(chain.from_iterable(zlist))
  zlist=list(map(str.strip, zlist))
  T=calculate_rank(zlist)
  M = [[0]*len(unique(zlist)) for _ in range(len(unique(zlist)))]
  for (i,j) in zip(T,T[1:]):
    M[i][j] += 1
  x_df=pd.DataFrame(M)
  np.fill_diagonal(x_df.values,0)
  x_df=pd.DataFrame(x_df.values/x_df.values.sum(axis=1)[:,None])
  x_df.columns=sorted(unique(zlist))
  x_df['index']=sorted(unique(zlist))
  x_df.set_index("index", inplace = True) 
  x_df.loc['convert',:]=0
  return(x_df)

#following function is used to return simulation path
def simulation(trans,n):
  sim=['']*n
  sim[0]= 'start'
  i=1
  while i<n:
    sim[i] = np.random.choice(trans.columns, 1, p=trans.loc[sim[i-1],:])[0]
    if sim[i]=='convert':
      break
    i=i+1       
  return sim[0:i+1]

def markov_chain(data_set,no_iteration=10,no_of_simulation=10000,alpha=5):
  import_dataset_v1=data_set.copy()
  import_dataset_v1=(import_dataset_v1.reindex(import_dataset_v1.index.repeat(import_dataset_v1.conversions))).reset_index()
  import_dataset_v1['conversions']=1
  import_dataset_v1=import_dataset_v1[['path','conversions']]
  import_dataset=(import_dataset_v1.groupby(['path']).sum()).reset_index()
  import_dataset['probability']=import_dataset['conversions']/import_dataset['conversions'].sum()
  final=pd.DataFrame()    
  for k in range(0,no_iteration):
    import_data=pd.DataFrame({'path':np.random.choice(import_dataset['path'],size=import_dataset['conversions'].sum(),p=import_dataset['probability'],replace=True)})
    import_data['conversions']=1                           
    tr_matrix=transition_matrix_func(import_data)
    tr_matrix.to_csv('transition_matrix.csv')
    channel_only = list(filter(lambda k0: k0 not in ['start','convert'], tr_matrix.columns)) 
    ga_ex=pd.DataFrame()
    tr_mat=tr_matrix.copy()
    p=[] 
    i=0
    while i < no_of_simulation:
      p.append(unique(simulation(tr_mat,1000)))
      i=i+1
    path=list(itertools.chain.from_iterable(p))
    counter=collections.Counter(path) 
    df=pd.DataFrame({'path':list(counter.keys()),'count':list(counter.values())})
    df=df[['path','count']]
    ga_ex=ga_ex.append(df,ignore_index=True) 
    df1=(pd.DataFrame(ga_ex.groupby(['path'])[['count']].sum())).reset_index()
    df1['removal_effects']=df1['count']/len(path)
    #df1['removal_effects']=df1['count']/sum(df1['count'][df1['path']=='convert'])
    df1=df1[df1['path'].isin(channel_only)]
    df1['ass_conversion']=df1['removal_effects']/sum(df1['removal_effects'])
    df1['ass_conversion']=df1['ass_conversion']*sum(import_dataset['conversions']) 
    final=final.append(df1,ignore_index=True)
      
  #H0: u=0
  #H1: u>0
    
  unique_channel=unique(final['path'])
  #final=(pd.DataFrame(final.groupby(['path'])[['ass_conversion']].mean())).reset_index()
  final_df=pd.DataFrame()  
  for i in range(0,len(unique_channel)):
    x=(final['ass_conversion'][final['path']==unique_channel[i]]).values
    final_df.loc[i,0]=unique_channel[i]
    final_df.loc[i,1]=x.mean() 
    #v=stats.ttest_1samp(x,0)
    #final_df.loc[i,2]=v[1]/2
    #if v[1]/2<=alpha/100:
    #  final_df.loc[i,3]=str(100-alpha)+'% statistically confidence'
    #else:
    #  final_df.loc[i,3]=str(100-alpha)+'% statistically not confidence'
    #final_df.loc[i,4]=len(x)
    #final_df.loc[i,5]=statistics.stdev(x)
    #final_df.loc[i,6]=v[0]
  final_df.columns=['channel','ass_conversion']
   #'p_value','confidence_status','frequency','standard_deviation','t_statistics']       
  final_df['ass_conversion']=sum(import_dataset['conversions']) *final_df['ass_conversion'] /sum(final_df['ass_conversion']) 
  return final_df,final

#calculation of the Markov model
dataset = finalData.loc[:, ['sourceMediumCampaignPath','totalConversions']]
dataset = dataset.rename(columns={'sourceMediumCampaignPath':'path', 'totalConversions':'conversions'})
import_dataset=dataset
markovData, markovDataset = markov_chain(import_dataset,no_iteration=10,no_of_simulation=10000,alpha=5)
print(markovData)
markovDataList = markovData.values.tolist()
numberOfConversion = sum([conversion[1] for conversion in markovDataList])

for value in markovDataList:
  key = value[0] + '; markov'
  values = value[0].split('/')
  result[key] = {
  'Source' : values[0],
  'Medium' : values[1],
  'Campaign' : values[2],
  'totalConversions' : value[1],
  'totalValue' : value[1]/numberOfConversion * conversionValueTotal,
  'Model' : 'markov'
    } 

#import counted data into CSV
toCSV = []
for key, value in result.items():
  toCSV.append(value)
header = toCSV[0].keys()
try:
   with open('output'+str(datetime.date.today())+'.csv',mode='w',encoding='utf8',newline='') as output_to_csv:
       dict_csv_writer = csv.DictWriter(output_to_csv, fieldnames=header,dialect='excel')
       dict_csv_writer.writeheader()
       dict_csv_writer.writerows(toCSV)
   print('\nData exported to csv succesfully and sample data')
except IOError as io:
    print('\n',io)

'''
#APP_DIR = os.path.dirname(os.path.realpath(__file__))
#sys.path.insert(0, APP_DIR)
#from model.FirstClick import FirstClick
#from model.Linear import Linear
#from model.Markov import Markov
#data = cursor.fetchall('SELECT * FROM dbo.Atrtributionmodels')
#result = []
#models = [FirstClick(), Linear(), Markov()]
#for model in models:
#    result += model.solve(data)
'''