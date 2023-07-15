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
from scipy.stats import pearsonr, spearmanr
# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

#turn of warnings 
import warnings
warnings.filterwarnings("ignore")


#---------- EXPLORATION FUNCTIONS ---------------

    

#------ Will use for FINAL REPORT------

def california_county(train):
    fig, ax = plt.subplots(figsize = (7,5))
    sns.scatterplot(data=train,x=train['longitude'],
                y=train['latitude'], zorder=1,hue='county')
plt.show()

#Question 1 Bathrooms
def bath_box(train):
    train_sample = train.sample(n=3017)
    features = ['bathrooms']
    
    for feature in features:
        sns.set(rc={'figure.figsize': (12, 12)})
        
        sns.boxplot(x=feature, y="tax_value", data=train_sample, hue='county')
        plt.title('Bathrooms vs County')
        
def bath_bar():    
    train_sample = train.sample(n=3017)
    # Visualizing bathrooms by county
    bath = sns.countplot(data=train_sample, x='bathrooms', hue='county')
    
    # Access the legend object
    legend = bath.legend()
    bath.set_xlabel('Bathrooms')
    bath.set_ylabel('Tax Value')
    plt.title('Bathrooms vs Tax Value')
                          
     # Add count numbers on bars
    for p in bath.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()    
        count = int(height)  # Get the count value
        offset = width * 0.02  # Adjust the offset percentage as needed
        bath.annotate(format(count, ',d'), (x + width / 2., y + height), ha='center', va='center', xytext=(0, 5), textcoords='offset points') 
    plt.show()

    
def bath_stat(train, validate, test):
    alpha = 0.05
    train_r, train_p = pearsonr(train.bathrooms, train.tax_value)
    validate_r, validate_p = pearsonr(validate.bathrooms, validate.tax_value)
    test_r, test_p = pearsonr(test.bathrooms, test.tax_value)
    print('train_r:', train_r)
    print('train_p:',train_p)
    print('validate_r:', validate_r)
    print('validate_p:', validate_p)
    print(f'The p-value is less than the alpha: {validate_p < alpha}')
    if validate_p < alpha:
        print('Outcome: We reject the null')
    else:
        print("Outcome: We fail to reject the null")
        
        
    

        
#Question 2 Sqft
def sqft_scat(train):
    sns.set(rc={'figure.figsize': (12, 12)})
    train_sample = train.sample(n=3017)
    sns.scatterplot(x= 'sqft', y="tax_value", data=train_sample, hue = 'county')
    plt.xlabel('Tax Value')
    plt.ylabel('SFH Square Feet ')
    plt.title('Tax Value vs. Square Feet')
    plt.show()
    
def sqft_stat(train, validate, test):
    alpha = 0.05
    train_r, train_p = spearmanr(train.sqft, train.tax_value)
    validate_r, validate_p = spearmanr(validate.sqft, validate.tax_value)
    test_r, test_p = spearmanr(test.sqft, test.tax_value)
    print('train_r:', train_r)
    print('train_p:',train_p)
    print('validate_r:', validate_r)
    print('validate_p:', validate_p)
    print('test_r:', test_r)
    print('test_p:', test_p)
    print(f'The p-value is less than the alpha: {test_p < alpha}')
    if test_p < alpha:
        print('Outcome: We reject the null')
    else:
        print("Outcome: We fail to reject the null")
        
        
        
# Question 3   Bedrooms     
def bed_in_box(train):
    train_sample = train.sample(n=3017)
    features = ['bedrooms']
    
    for feature in features:
        sns.set(rc={'figure.figsize': (12, 12)})
        
        sns.boxplot(x=feature, y="tax_value", data=train_sample, hue='county')
        plt.title('Bedrooms vs County')
        
def bed_in_bar(train):
    train_sample = train.sample(n=3017)
    x = ['bedrooms']  # Assuming you want to display the count for the 'bedrooms' feature
    
    # Visualizing the bathrooms by county
    br = sns.countplot(data=train_sample, x='bedrooms', hue='county')
    
    # Access the legend object
    legend = br.legend()
    
    br.set_xlabel('Bedrooms')
    br.set_ylabel('Tax Value')
    plt.title('Bedrooms vs Tax Value')
    
    # Add count numbers on bars
    for p in br.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()    
        count = int(height)  # Get the count value
        offset = width * 0.02  # Adjust the offset percentage as needed
        br.annotate(format(count, ',d'), (x + width / 2., y + height), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.show()
    
def plot_bed_pairs(train):
    train_sample = train.sample(n=3017)
    sns.lmplot(x='bedrooms', y="tax_value", data=train_sample, hue='county', line_kws={'color': 'red'})
    
#Bedrooms stat
def bedrooms_stat(train, validate, test):
    alpha = 0.05
    train_r, train_p = pearsonr(train.bedrooms, train.tax_value)
    validate_r, validate_p = pearsonr(validate.bedrooms, validate.tax_value)
    test_r, test_p = pearsonr(test.bedrooms, test.tax_value)
    print('train_r:', train_r)
    print('train_p:',train_p)
    print('validate_r:', validate_r)
    print('validate_p:', validate_p)
    print(f'The p-value is less than the alpha: {validate_p < alpha}')
    if validate_p < alpha:
        print('Outcome: We reject the null')
    else:
        print("Outcome: We fail to reject the null")
    
        
    #Question 4 Lot size
def lot_scat(train):
    sns.set(rc={'figure.figsize': (12, 12)})
    train_sample = train.sample(n=3017)
    sns.scatterplot(x='tax_value', y='lot_size', data=train_sample, hue='county')
    plt.xlabel('Tax Value')
    plt.ylabel('Lot Size')
    plt.title('Tax Value vs. Lot Size')
    plt.show()


def lot_stat(train, validate, test):
    alpha = 0.05
    train_r, train_p = spearmanr(train.lot_size, train.tax_value)
    validate_r, validate_p = spearmanr(validate.lot_size, validate.tax_value)
    test_r, test_p = spearmanr(test.lot_size, test.tax_value)
    print('train_r:', train_r)
    print('train_p:',train_p)
    print('validate_r:', validate_r)
    print('validate_p:', validate_p)
    print('test_r:', test_r)
    print('test_p:', test_p)
    print(f'The p-value is less than the alpha: {test_p < alpha}')
    if test_p < alpha:
        print('Outcome: We reject the null')
    else:
        print("Outcome: We fail to reject the null")
    
    
# EXPLORATION THAT DID NOT MAKE THE FINAL REPORT

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

def plot_categorical_and_continuous_vars(train):
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
    
# # call function with minmax
# mm_scaler = MinMaxScaler(feature_range=(0, 1))
# visualize_scaler(scaler=mm_scaler, df=train, features_to_scale=to_scale, bins=50)   

# # call function with standardscaler
# standard_scaler = StandardScaler()
# visualize_scaler(scaler=standard_scaler, df=train, features_to_scale=to_scale, bins=50)

# # call function with robustscaler
# r_scaler = RobustScaler()
# visualize_scaler(scaler=r_scaler, df=train, features_to_scale=to_scale, bins=50)


# qt_scaler = QuantileTransformer()
# visualize_scaler(scaler=qt_scaler, df=train, features_to_scale=to_scale, bins=50)



#------------ MODELING FUNCTIONS -----------
def baseline(y_train, y_validate):
    #  y_train and y_validate to be dataframes to append the new metric columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # Baseline for mean 
    # 1. Predict tax_value_pred_mean  make columns for train and validate
    tax_value_pred_mean = y_train.tax_value.mean()
    y_train['tax_value_pred_mean'] = tax_value_pred_mean
    y_validate['tax_value_pred_mean'] = tax_value_pred_mean 

  # 3. RMSE of tax_value_pred_mean
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_mean) ** (.5)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_mean) ** (.5)

    # create a df to easily view results of models
    metric_df = pd.DataFrame(data = [
        {
            'model': "mean_baseline",
            'RMSE_train': rmse_train,
            'RMSE_validate': rmse_validate,
            "R2_validate": explained_variance_score(y_validate.tax_value, y_validate.tax_value_pred_mean)
        }
    ])

    return y_train, y_validate, metric_df



def ols_lasso_tweedie(X_train, X_validate, y_train, y_validate, metric_df):
    ''' This function'''

    # make and fit OLS model
    lm = LinearRegression()

    OLSmodel = lm.fit(X_train, y_train.tax_value)

    # make a prediction and save it to the y_train
    y_train['tax_value_pred_ols'] = lm.predict(X_train)

    #evaluate RMSE
    rmse_train_ols = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_ols) ** .5

    # predict validate
    y_validate['tax_value_pred_ols'] = lm.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_ols = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_ols) ** .5

    #append metric
    metric_df = metric_df.append({
        'model': 'ols',
        'RMSE_train': rmse_train_ols,
        'RMSE_validate': rmse_validate_ols,
        'R2_validate': explained_variance_score(y_validate.tax_value, y_validate.tax_value_pred_ols)    
    }, ignore_index=True)

    print(f"""RMSE for OLS using LinearRegression
        Training/In-Sample:  {rmse_train_ols:.2f} 
        Validation/Out-of-Sample: {rmse_validate_ols:.2f}\n""")


    
    # make and fit OLS model
    lars = LassoLars(alpha=0.03)

    Larsmodel = lars.fit(X_train, y_train.tax_value)

    # make a prediction and save it to the y_train
    y_train['tax_value_pred_lars'] = lars.predict(X_train)

    #evaluate RMSE
    rmse_train_lars = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars) ** .5

    # predict validate
    y_validate['tax_value_pred_lars'] = lars.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_lars = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lars) ** .5

    #append metric
    metric_df = metric_df.append({
        'model': 'lasso_alpha0.03',
        'RMSE_train': rmse_train_lars,
        'RMSE_validate': rmse_validate_lars,
        'R2_validate': explained_variance_score(y_validate.tax_value, y_validate.tax_value_pred_lars)    
    }, ignore_index=True)

    print(f"""RMSE for LassoLars
        Training/In-Sample:  {rmse_train_lars:.2f} 
        Validation/Out-of-Sample: {rmse_validate_lars:.2f}\n""")


    # make and fit OLS model
    tr = TweedieRegressor(power=1, alpha=1.0)

    Tweediemodel = tr.fit(X_train, y_train.tax_value)

    # make a prediction and save it to the y_train
    y_train['tax_value_pred_tweedie'] = tr.predict(X_train)

    #evaluate RMSE
    rmse_train_tweedie = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_tweedie) ** .5

    # predict validate
    y_validate['tax_value_pred_tweedie'] = tr.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_tweedie = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_tweedie) ** .5

    # append metric
    metric_df = metric_df.append({
        'model': 'tweedie_power1_alpha1.0',
        'RMSE_train': rmse_train_tweedie,
        'RMSE_validate': rmse_validate_tweedie,
        'R2_validate': explained_variance_score(y_validate.tax_value, y_validate.tax_value_pred_tweedie)    
    }, ignore_index=True)

    print(f"""RMSE for TweedieRegressor
        Training/In-Sample:  {rmse_train_tweedie:.2f} 
        Validation/Out-of-Sample: {rmse_validate_tweedie:.2f}\n""")

    return y_train, y_validate, metric_df


def lasso_test_model(X_train, y_train, X_test, y_test):
    '''Switched models due to time constraints.This function fits the Lasso Model on train and predicts for test data.'''

    # MAKE and FIT Lasso+Lars
    lars = LassoLars(alpha = 0.03)
    LarsModel = lars.fit(X_train, y_train.tax_value)

    # predict with test data
    y_test_pred = lars.predict(X_test)

    # evaluate with RMSE
    LassoLars_rmse_test = mean_squared_error(y_test, y_test_pred) ** .5

    # calculate explained variance

    r2_test = explained_variance_score(y_test, y_test_pred)

    print(f"""RMSE for Lasso+Lars:
    _____________________________________________      
    Test Performance: {LassoLars_rmse_test:.2f}
    Test Explained Variance: {r2_test:.3f}
    Baseline: {y_train.tax_value.mean():.2f}""")
