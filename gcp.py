#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gcsfs #2021.11.1
import pandas as pd # '1.3.5'
import google.cloud.bigquery # 2.31.0
import numpy as np
from re import search
from datetime import datetime
from google.cloud import storage


def get_data(query):
    import google.auth
    from google.cloud import bigquery
    from google.cloud import bigquery_storage
    credentials, your_project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    bqclient = bigquery.Client(credentials=credentials, project=your_project_id,)
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

    your_query = query

    dataframe = (
        bqclient.query(your_query)
            .result()
            .to_dataframe(
                bqstorage_client=bqstorageclient,
                progress_bar_type='tqdm_notebook',))
    return dataframe

def move_blob(bucket_name, blob_name, destination_bucket_name, destination_blob_name):
    """
    Moves a blob from one bucket to another with a new name.
    
    Parameters
    ----------
    bucket_name : google.cloud.storage.client.Client
        Describes the bucket name of origin.
    blob_name : string
        Describes the name of the blob being moved.
    destination_bucket_name : string
        Describes the bucket name where the object is to be moved.
    destination_blob_name : string
        Describes the blob name where the object is to be moved.

    Returns
    -------
    None.
    
    """
    storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    blob_copy = source_bucket.copy_blob(
        source_blob, destination_bucket, destination_blob_name
    )
    source_bucket.delete_blob(blob_name)

    print(
        "Blob {} in bucket {} moved to blob {} in bucket {}.".format(
            source_blob.name,
            source_bucket.name,
            blob_copy.name,
            destination_bucket.name,
        )
    )

def join_files():
    """join the current week returns file with the produt hierarchy file from bigguqery..
    
    Parameters
    ----------

    Returns
    -------
    pandas.core.frame.DataFrame
    
    """
    df = pd.merge(new_returns_file, hierarchy_file, left_on = new_reutrns_product_key, right_on = hierarchy_product_key, how = 'left')
    return df

def clean_bigquery(df):
    """replace bigquery hierarhcy_file columns title containing "_" to " " for consistency..
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Contains file from bigguqery.
    Returns
    -------
    pandas.core.frame.DataFrame
    
    """
    for i in range(0,len(df.columns)):
        if(search("_", df.columns[i])):
            df = df.rename(columns= {df.columns[i]: df.columns[i].replace("_"," ")})
    return df

def capitalize_title(df):
    """Capitalizes the first character of each column title word for consistency in appending in the future..
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Contains the data.
    Returns
    -------
    pandas.core.frame.DataFrame
    
    """
    for i in range(0,len(df.columns)):
        df = df.rename(columns={df.columns[i]: df.columns[i].title()})
    return df

def bigquery_title_adapt(df):
    """Replacing non-bigquery accepted characters to appropriate title: 
        " ", "#" , "(" , ")" , "/" replaced as bigquery does not allow special characters and empty spaces in column titles..
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Contains data.
    Returns
    -------
    pandas.core.frame.DataFrame
    
    """
    for i in range(0,len(df.columns)):
        if search(" ",df.columns[i]):
                df = df.rename(columns={df.columns[i]: df.columns[i].replace(" ","_")})
        if  search("#",df.columns[i]):
                df = df.rename(columns={df.columns[i]: df.columns[i].replace("#","Number")})
        if  search(r"\(",df.columns[i]):
                df = df.rename(columns={df.columns[i]: df.columns[i].replace(r'(' ,"")})
        if  search(r"\)",df.columns[i]):
                df = df.rename(columns={df.columns[i]: df.columns[i].replace(r')' ,"")})
        if  search("/", df.columns[i]):
               df = df.rename(columns={df.columns[i]: df.columns[i].replace("/" ,"or")})
    return df

def Origin_Zip_trans(df):
    """Changing the zipcode column to string type: 
        Canada zipcode includes character values, integer/float does not apply..
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Contains  data.
    Returns
    -------
    pandas.core.frame.DataFrame
    
    """
    df['Origin_Zip'] = df['Origin_Zip'].astype(str)

    return df

def date_parse(df):
    """Parse date columns from string to pandas date format..
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Contains data to parse.
    Returns
    -------
    pandas.core.frame.DataFrame
    
    """
    date_cols = [

    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors = 'coerce')
        
    return df

def change_to_str(df):
    """replacing every column to string type in order to fit the bigquery requirements..
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Contains data.
    Returns
    -------
    pandas.core.frame.DataFrame
    
    """
    for i in range(0,len(df.columns)):
        df[df.columns[i]] = df[df.columns[i]].astype(str)
    return df


def coerce_format(df):
    """changing columns that contain numeric observations into numeric or float data types for bigquery requirements:
        Along with certain variables, all variables with column name containing "" is changed to integer
        First, all of the integer variables are changed to float then to integer because Python does not read str --> int directly..
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Contain sdata.
    Returns
    -------
    pandas.core.frame.DataFrame
    
    """
    
    float_cols = [

    ]
    
    int_cols = [

    ]

    for col in float_cols:
        df[col] = df[col].astype(float)

    for col in int_cols:
        df[col] = df[col].astype(int)    
    
    for i in range(0,len(df.columns)):
        if search("Cd",df.columns[i]):
            df[df.columns[i]] = pd.to_numeric(df[df.columns[i]]).astype('int').fillna(-1)
            
    return df
            
def to_bq(df):
    project = 
    dataset = 
    table = 
    
    client = google.cloud.bigquery.Client()

    dataset_ref = client.dataset(dataset, project = project)
    table_ref = dataset_ref.table(table)  

    job_config = google.cloud.bigquery.LoadJobConfig( 
     schema=[ 
         google.cloud.bigquery.SchemaField("Order_Number", google.cloud.bigquery.enums.SqlTypeNames.STRING),
 
     ],
     autodetect = False,
     write_disposition="WRITE_APPEND",
    )
    print(f'Creating table {project}:{dataset}.{table}')
    client.load_table_from_dataframe(df, table_ref, job_config=job_config).result()
    
    return None

def main(event, context):

    print('Event ID: {}'.format(context.event_id))
    print('Event type: {}'.format(context.event_type))
    print('Bucket: {}'.format(event['bucket']))
    print('File: {}'.format(event['name']))
    print('Metageneration: {}'.format(event['metageneration']))
    print('Created: {}'.format(event['timeCreated']))
    print('Updated: {}'.format(event['updated']))

    # read
    file_path = 'gs://'
    df = pd.read_csv(file_path)

    #load bq query (hierarchy)
    query =  """SELECT * FROM """ 
                    
    hierarchy = get_data(query)               

    #merge files
    df = join_files(df, df)

    #replace _ in the  table in order to capitalize first letter for consistency in title in the future
    df = clean_bigquery(df)

    #capitalize first letter of each word in column title for consistency
    df = capitalize_title(df)

    #replacing space, #, (, and ) as the bigquery column titles do not process them
    df = bigquery_title_adapt(df)
    
    #change "nan" to -1
    df = df.fillna(-1)

    #change everything to strings
    df = change_to_str(df) 
    
    #coerce format appropriately
    df = coerce_format(df)
    
    #change date columns to datetime data types
    df = date_parse(df)

    #create bq table
    to_bq(df)
    
    #move the file to archive
    move_blob('',event['name'], 
              '','archive/'+event['name'])

    return None


# In[35]:


main(2,2)

