#!/usr/bin/env python
# coding: utf-8

# ### IMPORTS 

# In[1]:


#imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

#functions
import tqdm
import logging
import google.auth
import pandas as pd
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import bigquery_storage

def config_bq_gcs(auth="https://www.googleapis.com/auth/cloud-platform"):
    """
    Configuring bq,gcs API using Service Account

    Parameters
    ----------
    auth : string, optional
        Auth string to get default permissions/scope for GCP APIs. The default is "https://www.googleapis.com/auth/cloud-platform".

    Returns
    -------
    Configurations to interact with BQ & GCS

    """

    credentials, your_project_id = google.auth.default(
        scopes=[auth])
    client_bq = bigquery.Client(
        credentials=credentials, project=your_project_id,)
    logging.info("BQ client config successful")
    client_gcs = storage.Client()
    logging.info("gcs client config successful")
    client_bq_storage = bigquery_storage.BigQueryReadClient(
        credentials=credentials)
    logging.info("BQ storage client config successful")
    return(client_bq, client_gcs, client_bq_storage)

def bq_import(query):
    """
    Export data from BQ to DataFrame

    Parameters
    ----------
    query : string 
        Query to be executed.

    Returns
    -------
    DataFrame with BQ Query results

    """
    try:
        df = (
            client_bq.query(query)
            .result()
            .to_dataframe(
                bqstorage_client=client_bq_storage,
                progress_bar_type='tqdm',)
        )

        logging.info("BQ export successful")
    except Exception as e:
        logging.error(
            'Received error while exporting BQ table: \n{}'.format(e))

    return df

client_bq, client_gcs, client_bq_storage = config_bq_gcs()


# ### TRAIN

# In[2]:


for tbleName in ['']:
    sample_query = f"""SELECT * from {tbleName} """
    train_total = bq_import(sample_query)  


# In[3]:


train_total = train_total.sort_values(by =[""], axis = 0)


# In[4]:


train_total = train_total.reset_index()
train_total = train_total.drop("index", axis =1)


# In[5]:


train_pos = train_total.iloc[0:1207713,:]
train_neg = train_total.iloc[1207713:6038565,:]


# In[6]:


train_neg = train_neg.sample(n=1207713)


# In[7]:


train_pos['class'] = int(1)
train_neg['class'] = int(0)


# In[8]:


train = pd.concat([train_pos,train_neg])


# In[10]:


import pytz
from datetime import datetime
d = datetime(2022,1,10,0,0,0,0, pytz.UTC)
train[''] = d - train['']
train[''] = d- train['']


# In[11]:


train = train.drop(["","","",""], axis = 1)
train[['','']] = train[['','']].astype(int)
train[["","","","",""]] = train[["","","","",""]].astype('category')


# In[12]:


train = train.fillna(0)


# In[30]:


train_num_scale = train.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]]

train_num_scale_2 = train.iloc[:,17:185]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_num_scale)
train_num_scale = scaler.transform(train_num_scale)

train.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]] = train_num_scale
train.iloc[:,17:185] = train_num_scale_2


# In[31]:


y_train = train['class']
x_train = train.drop('class', axis = 1)
x_train = x_train.fillna(0)
x_train = x_train.drop('', axis = 1)


# In[36]:


import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Hyper Parameter Tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_features': ['sqrt'],
    'max_depth': [10, 50],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2, 4, 8]
}

rf = sklearn.ensemble.RandomForestClassifier()
rf_grid = sklearn.model_selection.GridSearchCV(
    estimator = rf,
    param_grid = param_grid,
    cv = 3,
    verbose=5,
    n_jobs = 20
)

rf_grid.fit(x_train, y_train)

best_params = rf_grid.best_params_
model = rf_grid.best_estimator_


# In[ ]:


import pycaret.classification
data = x_train # training data after cleaning/feature engineering
data['response'] = y_train # response variable

clf1 = pycaret.classification.setup(data = x_train, target = 'response')

# compare models
best = pycaret.classification.compare_models()


# ### TEST

# In[21]:


for tbleName in ['']:
    sample_query = f"""SELECT * from {tbleName} """
    test = bq_import(sample_query)  


# In[22]:


import pytz
from datetime import datetime
d = datetime(2022,1,10,0,0,0,0, pytz.UTC)
test[''] = d - test['']
test[''] = d- test['']


# In[23]:


test = test.drop(["","","",""], axis = 1)
test[['','']] = test[['','']].astype(int)
test[["","","",""]] = test[["","","",""]].astype('category')


# In[24]:


test = test.fillna(0)


# In[25]:


test_num_scale = test.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                                41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,
                                77,78,79,80,81,82,83,84]]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(test_num_scale)
test_num_scale = scaler.transform(test_num_scale)

test.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                                41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,
                                77,78,79,80,81,82,83,84]] = test_num_scale


# In[26]:


x_test = test.drop('', axis = 1)


# In[ ]:


x_test.info()


# In[27]:


from tqdm import tqdm_notebook as tqdm

y_pred = []

for df_chunk in tqdm(np.array_split(x_test, 1000)):
    y_pred = np.append(y_pred, model.predict(df_chunk))


# In[28]:


result = pd.DataFrame()
result['predict'] = y_pred
result[''] = test['']

