import pandas as pd
import numpy as np
import os
from env import hostname, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings
warnings.filterwarnings("ignore")

def get_connection(db, user=user, hostname =hostname, password=password):
    return f'mysql+pymysql://{user}:{password}@{hostname}/{db}'

def zillow_data():
    #Save before editing so that we still have the original on hand
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        return get_zillow_data()
# def get_zillow_data():
#     '''
#     This function reads in the Zillow data from the Codeup db
#     and returns a pandas DataFrame with all columns.
#     '''
    
#     filename = 'zillow.csv'

#     if os.path.isfile(filename):
#         return pd.read_csv(filename)
#     else:     
#         sql = '''
#                 SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
#                 FROM  properties_2017
#                 WHERE propertylandusetypeid = 261
#                 '''
#         df.to_csv(filename, index=False)
#         df = pd.read_sql(sql, get_connection('zillow')) 
#         return df
def get_zillow_data():
    #this sql Query
    sql_query = """SELECT
    bedroomcnt,
    bathroomcnt,
    calculatedfinishedsquarefeet,
    taxvaluedollarcnt,
    fips
FROM
    properties_2017
        JOIN
    predictions_2017 USING (propertylandusetypeid)
WHERE
    propertylandusetypeid = 261
"""
    df = pd.read_sql(sql_query, get_connection('zillow'))
    return df
