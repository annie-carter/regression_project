#python libraries
import pandas as pd
import numpy as np
import os

#import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns 


#import 
from env import hostname, user, password
import wrangle as w

#Import scikit-learn 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

#turn of warnings 
import warnings
warnings.filterwarnings("ignore")


#---------- EXPLORE ---------------

    


def california_county(train):
    fig, ax = plt.subplots(figsize = (7,5))
    sns.scatterplot(data=train,x=train['longitude'],
                y=train['latitude'], zorder=1,hue='county')
plt.show()

def pairplot(train):
    sns.pairplot(train, hue = 'county')
    plt.title('Pairplot of Zillow features')
    plt.show()
    
def heatmap(train):
    #Visualizing correlation data with Heat Map
    plt.figure(figsize=(25,20))
    sns.heatmap(train.corr(), cmap='Blues', center=0, annot=True)
    plt.show()
    
#random sample of 3017 which is apprx 10% of training data 
def plot_variable_pairs(train):
    train_sample = train.sample(n=3017)
    features = ['bedrooms', 'bathrooms', 'sqft', 'lot_size']
    for feature in features:
        sns.lmplot(x=feature, y="tax_value", data=train_sample, hue='county', line_kws={'color': 'red'})        



def plot_categorical_and_continuous_vars():
    train_sample = train.sample(n=3017)
    features = ['bedrooms', 'bathrooms', 'sqft', 'lot_size',]
    
    for feature in features:
        sns.set(rc={'figure.figsize': (30, 15)})
        fig, axes = plt.subplots(2, 2)
        
        sns.boxplot(x=feature, y="tax_value", data=train_sample, hue='county', ax=axes[0, 0])
        axes[0, 0].set_title('Boxplot')
        
        sns.barplot(x=feature, y="tax_value", data=train_sample, hue='county', ax=axes[0, 1])
        axes[0, 1].set_title('Barplot')
        
        sns.violinplot(x=feature, y="tax_value", data=train_sample, hue='county', ax=axes[1, 0])
        axes[1, 0].set_title('Violinplot')
        
        sns.scatterplot(x=feature, y="tax_value", data=train_sample, hue='county', ax=axes[1, 1])
        axes[1, 1].set_title('Scatterplot')
        
    plt.tight_layout()
plot_categorical_and_continuous_vars()



#--------- SCALING FUNCTIONS--------
def visualize_scaler(scaler, df, features_to_scale, bins=50):
    # Create subplot structure
    fig, axs = plt.subplots(len(features_to_scale), 2, figsize=(12, 12))

    # Copy the df for scaling
    df_scaled = df.copy()

    # Fit and transform the df
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Plot the pre-scaled data next to the post-scaled data in one row of a subplot
    for (ax1, ax2), feature in zip(axs, features_to_scale):
        ax1.hist(df[feature], bins=bins)
        ax1.set(title=f'{feature} before scaling', xlabel=feature, ylabel='count')
        ax2.hist(df_scaled[feature], bins=bins)
        ax2.set(title=f'{feature} after scaling with {scaler.__class__.__name__}', xlabel=feature, ylabel='count')
    plt.tight_layout()
    
# call function with minmax
mm_scaler = MinMaxScaler(feature_range=(0, 1))
visualize_scaler(scaler=mm_scaler, df=train, features_to_scale=to_scale, bins=50)   

# call function with standardscaler
standard_scaler = StandardScaler()
visualize_scaler(scaler=standard_scaler, df=train, features_to_scale=to_scale, bins=50)

# call function with robustscaler
r_scaler = RobustScaler()
visualize_scaler(scaler=r_scaler, df=train, features_to_scale=to_scale, bins=50)


qt_scaler = QuantileTransformer()
visualize_scaler(scaler=qt_scaler, df=train, features_to_scale=to_scale, bins=50)



#------------ MODELING-----------
def ols_lasso_tweedie(X_train, X_validate, y_train, y_validate, metric_df):
    ''' This function'''

    # make and fit OLS model
    lm = LinearRegression()

    OLSmodel = lm.fit(X_train, y_train.value)

    # make a prediction and save it to the y_train
    y_train['value_pred_ols'] = lm.predict(X_train)

    #evaluate RMSE
    rmse_train_ols = mean_squared_error(y_train.value, y_train.value_pred_ols) ** .5

    # predict validate
    y_validate['value_pred_ols'] = lm.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_ols = mean_squared_error(y_validate.value, y_validate.value_pred_ols) ** .5

    #append metric
    metric_df = metric_df.append({
        'model': 'ols',
        'RMSE_train': rmse_train_ols,
        'RMSE_validate': rmse_validate_ols,
        'R2_validate': explained_variance_score(y_validate.value, y_validate.value_pred_ols)    
    }, ignore_index=True)

    print(f"""RMSE for OLS using LinearRegression
        Training/In-Sample:  {rmse_train_ols:.2f} 
        Validation/Out-of-Sample: {rmse_validate_ols:.2f}\n""")


    
    # make and fit OLS model
    lars = LassoLars(alpha=0.03)

    Larsmodel = lars.fit(X_train, y_train.value)

    # make a prediction and save it to the y_train
    y_train['value_pred_lars'] = lars.predict(X_train)

    #evaluate RMSE
    rmse_train_lars = mean_squared_error(y_train.value, y_train.value_pred_lars) ** .5

    # predict validate
    y_validate['value_pred_lars'] = lars.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_lars = mean_squared_error(y_validate.value, y_validate.value_pred_lars) ** .5

    #append metric
    metric_df = metric_df.append({
        'model': 'lasso_alpha0.03',
        'RMSE_train': rmse_train_lars,
        'RMSE_validate': rmse_validate_lars,
        'R2_validate': explained_variance_score(y_validate.value, y_validate.value_pred_lars)    
    }, ignore_index=True)

    print(f"""RMSE for LassoLars
        Training/In-Sample:  {rmse_train_lars:.2f} 
        Validation/Out-of-Sample: {rmse_validate_lars:.2f}\n""")


    # make and fit OLS model
    tr = TweedieRegressor(power=1, alpha=1.0)

    Tweediemodel = tr.fit(X_train, y_train.value)

    # make a prediction and save it to the y_train
    y_train['value_pred_tweedie'] = tr.predict(X_train)

    #evaluate RMSE
    rmse_train_tweedie = mean_squared_error(y_train.value, y_train.value_pred_tweedie) ** .5

    # predict validate
    y_validate['value_pred_tweedie'] = tr.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_tweedie = mean_squared_error(y_validate.value, y_validate.value_pred_tweedie) ** .5

    # append metric
    metric_df = metric_df.append({
        'model': 'tweedie_power1_alpha1.0',
        'RMSE_train': rmse_train_tweedie,
        'RMSE_validate': rmse_validate_tweedie,
        'R2_validate': explained_variance_score(y_validate.value, y_validate.value_pred_tweedie)    
    }, ignore_index=True)

    print(f"""RMSE for TweedieRegressor
        Training/In-Sample:  {rmse_train_tweedie:.2f} 
        Validation/Out-of-Sample: {rmse_validate_tweedie:.2f}\n""")

    return y_train, y_validate, metric_df


#Will use for final 
def bath_box():
    train_sample = train.sample(n=3017)
    features = ['bathrooms']
    
    for feature in features:
        sns.set(rc={'figure.figsize': (12, 12)})
        
        sns.boxplot(x=feature, y="tax_value", data=train_sample, hue='county')
        plt.title('Bathrooms vs County')
        
def bed_in_box():
    train_sample = train.sample(n=3017)
    features = ['bedrooms']
    
    for feature in features:
        sns.set(rc={'figure.figsize': (12, 12)})
        
        sns.boxplot(x=feature, y="tax_value", data=train_sample, hue='county')
        plt.title('Bedrooms vs County')