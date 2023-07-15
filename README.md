# <bu>REGRESSION_PROJECT</bu>


## Project Description
This project predicts factors contributing to 2017 property tax assessed values for three California counties ( Orange County, Ventura, Los Angeles). The data collected from this project can help with future planning on ways to decrease property tax and improve loyalty to the Zillow brand.


## Project Goals
* This project goal is to develop an ML Regression model to accurately forecast the assessed values of Single Family Properties by leveraging property attributes. 
* Identify the crucial factors property tax assessed values of single family properties that had a transaction during 2017. 
* Present findings to lead data scientist and other Zillow stakeholders


## Initial Questions 
To guide our analysis, we initially posed the following questions:

1. Does number of bathrooms have a relationship to property tax value?
2. Is there a relationship between square feet and property tax value?
3. Does number of bedrooms have a relationship to property tax value?
4. Does the Lot Size have a significant impact on property tax value?


## Data Dictionary

There were ## columns in the initial data and ## columns after preparation; ### rows in the intial data and ### after preparation the central target column will be property_tax: 

| Original                    |   Target    |       Datatype          |       Definition             |
|-----------------------------|-------------|-------------------------|------------------------------|
|  taxvaluedollarcnt          |  tax_value  | 50283 non-null: float64 |  target variable             |


|                             |   Feature    |       Datatype         |     Definition               |
|-----------------------------|--------------|------------------------|------------------------------|
|  bedroomcnt                 |  bedrooms    | 50283 non-null: int64  | # of bedrooms                |
|  bathroomcnt                |  bathrooms   | 50283 non-null: int64  | # of bathrooms               |
|calculatedfinishedsquarefeet |  sqft        | 50283 non-null: int64  | # of square feet             |
|  yearbuilt                  |  year_built  | 50283 non-null: object | year house was built)        |
|  fips                       |  county      | 50283 non-null: object | County house located         |
|  lotsizesquarefeet          |  lot_size    | 50283 non-null: int64  | size of lot                  |
|  longitude                  |  longitude   | 50283 non-null: int64  | longitude line house located |
|  latitude                   |  latitude    | 50283 non-null: int64  | latitude line house located  |    


## Project Planning
### Planning
1. Clearly define the problem to be investigated, such as the impact  square feet on property assessed tax value.
2. Obtain the required data from the "Zillow.csv" database.
3. Create a comprehensive README.md file documenting all necessary information.
### Acquisition and Preparation
4. Develop the acquire.py and prepare.py scripts, which encompass functions for data acquisition, preparation, and data splitting.
5. Implement a .gitignore file to safeguard sensitive information and include files (e.g., env file) that require discretion.
### Exploratory Analysis
6. Create preliminary notebooks to conduct exploratory data analysis, generate informative visualizations, and perform relevant statistical tests (e.g.,t-test, chi-square test) utilizing Random Seed value 123 and alpha = .05.
### Modeling
7. Train and evaluate various models, such as Ordinary Least Squares (OLS) Linear Regression, Least Absolute Shrinkage and Selection Operator (LASSO),  Least Angle Regression (LARS), utilizing a Random Seed value of 123 and alpha= 1.0.
    * Train the models using the available data.
    * Validate the models to assess their performance.
    * Select the most effective model (e.g., Logistic Regression) for further testing.
### Product Delivery
8, Prepare a final notebook that integrates the best visuals, models, and pertinent data to present comprehensive insights.


## Instructions  to Reproduce the Final Project Notebook
To successfully run/reproduce the final project notebook, please follow these steps:
1. Read this README.md document to familiarize yourself with the project details and key findings.
2. Before proceeding, ensure that you have the necessary database credentials. Create or use an environment (env) file that includes the required credentials, such as username, password, and host. Make sure not to add your env file to the project repository, but .gitignore.
3. Clone the Regression_project repository from my GitHub or download the following files: aquire.py, prepare.py, and final_report.ipynb. You can find these files in the project repository.
4. Open the final_report.ipynb notebook in your preferred Jupyter Notebook environment or any compatible Python environment.
5. Ensure that all necessary libraries or dependent programs are installed. You may need to install additional packages if they are not already present in your environment.
6. Run the final_report.ipynb notebook to execute the project code and generate the results.

By following these instructions, you will be able to reproduce the analysis and review the project's final report. Feel free to explore the code, visualizations, and conclusions presented in the notebook.

## Initial Hypotheses 
Hypothesis 1 - Pearson R
* alpha = .05 
* H0 = Number of Bathrooms has no relationship with of property tax value
* Ha = Number of Bathrooms has a relationship with property tax value
* <b> Outcome: We reject the Null Hypothesis.</b>

Hypothesis 2 - Spearman R
* alpha = .05 
* H0 = Square Feet has no correlation with property tax value
* Ha = Square Feet is correlated to property tax value
* <b>Outcome: We reject the Null Hypothesis.</b>
    
Hypothesis 3 - Pearson R
* alpha = .05 
* H0 = Number of Bedrooms has no relationship with property tax value
* Ha = Number of Bedrooms has a relationship with on property tax value
* <b>Outcome: We reject the Null Hypothesis.</b>

Hypothesis 4 - Spearman R 
* alpha = .05 
* H0= Lot Size has no significant correlation with property tax value
* Ha= Lot Size is correlated  to property tax value
* <b>Outcome: We reject the Null Hypothesis.</b>
* Inconsistent findings due to extremely low correlation.

## Key findings
- After selecting four features and conducting data visualizations, scaling, and statistical testing using Pearson R and Spearman R, the correlation coefficients (r-values) between the features and the target variable in both the training and validation datasets show a positive relationship with small p-values.
- Across the three different counties, all features exhibit a correlation/relationship with property tax value ranging from very weak to moderate. Among these features, square footage demonstrates the strongest regression line.
- In Los Angeles County, there is a higher average of single-family homes with 6+ bedrooms, indicating higher tax values and larger lot sizes. Bedrooms have a lesser impact on property tax value compared to bathrooms. Ventura County has a scarcity of one-bedroom homes, while Orange County shows a distribution comparable to bedrooms and tax value.
- The rejection of null hypotheses for bedrooms, square footage, bathrooms, and lot size provides evidence of a significant relationship with property tax value.
- These findings confirm the significance of the correlations and validate the use of the Pearson R correlation test, indicating a somewhat normal distribution of the data.

## Conclusion
This project aimed to develop a machine learning regression model to forecast the assessed values of single-family properties using property attributes. The analysis revealed that all features across the three different counties showed varying correlations with property tax value, with square footage exhibiting the strongest relationship. Los Angeles County had a higher average of 6+ bedroom single-family homes with higher tax values and larger lot sizes. Bedrooms had a lesser impact on tax value compared to bathrooms, and Ventura County had fewer one-bedroom homes. The null hypotheses were rejected for bedrooms, square footage, bathrooms, and lot size, indicating significant relationships with property tax value. The best performing models were OLS and Lasso+Lars, with an RMSE of 275,079 and 278,281 respectively, outperforming the baseline. The chosen model for further analysis is Lasso+Lars (alpha = 0.03).
   
## Next Steps
Based on the findings, the following recommendations and next steps are proposed:

1. Conduct Polynomial Regression: To delve deeper into the relationship between the features and property tax values, it is recommended to perform Polynomial Regression analysis. This will provide additional insights into the non-linear relationships and potential higher-order interactions between the variables.

2. Create Dummy Variables for Counties: In order to capture the variations across different counties, it is advised to create dummy variables to split the data by counties. This will enable the development of separate models for each county, allowing for a more comprehensive examination of the unique factors influencing property tax values within each region.

3. Explore Additional Features: Further exploration of the dataset should include an investigation of additional features that may impact property tax values. Consideration can be given to factors such as neighborhood characteristics, proximity to amenities, or economic indicators to gain a more comprehensive understanding of the factors at play.

By pursuing these steps, a more comprehensive analysis can be achieved, providing valuable insights into the factors influencing property tax values across the Los Angeles, Ventura and Orange counties.

   
#### Recommendations 
- To improve prediction accuracy, it is recommended to aggregate at least three years of past data.
- During the data cleaning process, using the Interquartile Range (IQR) method with the 25th and 75th percentiles is advised to avoid overfitting and address the natural skewness in the data.
- In order to enhance the prediction model, incorporating the "bathbdcnt" column from the original zillow.csv dataset is recommended, as it eliminates redundancy and improves the accuracy of predictions.
- However, it is advisable to exclude variables such as "fireplace" and "basement" from the analysis due to a high number of null values, despite their significant correlations with tax value prediction.