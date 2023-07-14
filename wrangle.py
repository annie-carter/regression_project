import pandas as pd
import numpy as np
import os
from env import hostname, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,QuantileTransformer
import warnings
warnings.filterwarnings("ignore")


    #---------- ACQUIRE -----------

def get_connection(db, user=user, hostname =hostname, password=password):
    ''' The below functions were created to acquire the 2017 Zillow data from CodeUp database and make a SQL query to meet project goals 
        removing unnecessary information.
    '''
    return f'mysql+pymysql://{user}:{password}@{hostname}/{db}'


def get_zillow_data():
    # SQL Query
    filename = 'zillow.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        sql_query = '''
                    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, fips, lotsizesquarefeet, longitude, latitude
                    FROM properties_2017
                    JOIN predictions_2017 USING (parcelid)
                    JOIN propertylandusetype USING (propertylandusetypeid)
                    WHERE propertylandusetypeid = 261 AND transactiondate LIKE '2017%%';
                    '''
        df = pd.read_sql(sql_query, get_connection('zillow'))
        df.to_csv(filename, index=False)
        return df

    
  #-------------PREPARE DATA------------  
    
def prep_zillow(df):
    ''' 
     The below functions prepare the 2017 Zillow data by renaming the columns to enhance readability, remove nulls and duplicate 
    information.
    '''
    #change column names to be more readable
    df = df.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 'calculatedfinishedsquarefeet':'sqft','taxvaluedollarcnt':'tax_value', 'yearbuilt':'year_built', 'lotsizesquarefeet' : 'lot_size', 'fips':'county'})
    #drop null values 
    df = df.dropna()
    #drop duplicates
    df.drop_duplicates(inplace=True)
    return df

#--------- CLEAN DATA FUNCTIONS---------
    

def rename_county(df):
    ''' The below functions were created in regression excercises and will be aggregated to make a master clean_data function for final report
    '''
    df['county'] = df.county.map({6037.0: 'Los Angeles', 6059.0: 'Orange Cty', 6111.0: 'Ventura'})
    return df


def remove_nobed_nobath(df):
    df = df[(df.bedrooms != 0) & (df.bathrooms != 0) & (df.sqft >= 70)]
    return df

def remove_outliers(df):
    #eliminate outliers
    df = df[df.bathrooms <= 6]
    df = df[df.bedrooms <= 6]
    df = df[df.tax_value < 2_000_000]
    return df 

def dtype_zillow(df):
    # Convert bedrooms, bathrooms, and sqft columns to integers
    df['bedrooms'] = df['bedrooms'].astype(int)
    df['bathrooms'] = df['bathrooms'].astype(int)
    df['sqft'] = df['sqft'].astype(int)
    df['lot_size'] = df['lot_size'].astype(int)
    df['longitude'] = df['longitude'].astype(int)
    df['latitude'] = df['latitude'].astype(int)
    
    # Convert year_built and fips columns to integers and then to strings
    df['year_built'] = df['year_built'].astype(int).astype(str)   
    return df




def master_clean_zillow(df):
    df = rename_county(df)
    df = remove_nobed_nobath(df)
    df = remove_outliers(df)
    df = dtype_zillow(df)
    return df

#----------SPLIT DATA---------

def split_zillow(df):
    ''' The below functions were created in regression excercises and will be aggregated to make a master clean_data function for final 
        report
    '''
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)
    return train, validate, test


def print_train(train, validate, test):
    print(f'Train shape: {train.shape}')
    print(f'Validate shape: {validate.shape}')
    print(f'Test shape: {test.shape}')

def x_y_split(train, validate, test):
    X_train, y_train = train.select_dtypes('float').drop(columns='tax_value'), train.tax_value
    X_validate, y_validate = validate.select_dtypes('float').drop(columns='tax_value'),validate.tax_value
    X_test, y_test = test.select_dtypes('float').drop(columns='tax_value'), test.tax_value
    return X_train, y_train,X_validate,y_validate,X_test,y_test

def scaled_data(train, validate, test):
    '''This function takes in the train, validate, and test datasets removes the county columns so the data can be scaled.'''
    train = train.drop(columns='county') 
    validate = validate.drop(columns='county')
    test = test.drop(columns='county')
    return train, validate, test