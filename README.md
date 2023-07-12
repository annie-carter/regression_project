# REGRESSION_PROJECT


## Project Description
This project predicts factors contributing to 2017 property tax assessed values for three California counties ( Orange County, Ventura, Los Angeles). The data collected from this project can help with future planning on ways to decrease property tax and improve loyalty to the Zillow brand.


## Project Goals
* This project goal is to develop an ML Regression model to accurately forecast the assessed values of Single Family Properties by leveraging property attributes. 
* Identify the crucial factors property tax assessed values of single family properties that had a transaction during 2017. 
* Present findings to lead data scientist and other Zillow stakeholders


## Initial Questions 
To guide our analysis, we initially posed the following questions:

1. Does the County (fip) have a significant impact on property tax value?
2. Does lot size have a relationship to property tax value?
3. Is there a relationship between bedrooms and property tax value?


## Initial Hypotheses 
Hypothesis 1 
* alpha = .05 
* H0= California County is independent of property tax
* Ha= California County is dependent on property tax
* Outcome: We will accept or reject the Null Hypothesis.

Hypothesis 2 
* alpha = .05 
* H0 = Lot Size is independent of property tax 
* Ha = Lot Size is dependent on property tax
* Outcome: We will  accept or reject the Null Hypothesis.
    
Hypothesis 3 
* alpha = .05 
* H0 = Bedrooms is independent of property tax 
* Ha = Bedrooms is dependent on property tax
* Outcome: We will  accept or reject the Null Hypothesis.



## Data Dictionary

There were ## columns in the initial data and ## columns after preparation; ### rows in the intial data and ### after preparation the central target column will be property_tax: 

|   Targee    |       Datatype        |    Definition      |
|------------|-----------------------|:------------------: |
| property_tax  | ##### non-null: object |   (int)   |


|        Feature          |       Datatype        |     Definition        |
|-------------------------|-----------------------|-----------------------|
|lot_size 	      |##### non-null: object  | (integer)       |
|bedrooms	      |##### non-null: uint8   | (integer)       |
|bathrooms	      |##### non-null: uint8   | (integer)       |
|sqft	          |##### non-null: uint8   | (integer)       |
|lot_size 	      |##### non-null: object  | (integer)       |
|bedrooms	      |##### non-null: uint8   | (integer)       |
|bathrooms	      |##### non-null: uint8   | (integer)       |
|sqft	          |##### non-null: uint8   | (integer)       |

## Project Planning
### Planning
1. Clearly define the problem to be investigated, such as the impact lot size on property assessed tax value.
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


## Key findings/ Conclusion
After selecting three features and creating data visualizations, I chose the features that displayed visual significance and had a more significant relationship in _____chi-square_____ statistical testing to train the Regression Model which were California County and Lot Size. Among the tested models, ____________ emerged as the most effective model for predicting property assessed value, surpassing the baseline accuracy of ___% with a consistent accuracy of ____% across the train, validate, and test sets.
   
Hypothesis 1: California County and Property Tax Value
- Outcome: We ______ the Null Hypothesis, indicating that California County is dependent on property tax value.
- However, the relationship between California County and property tax is dependent, with a count of _###_ out of ###. It can be utilized in the modeling process.

Hypothesis 2: Lot Size and Property Tax Value
- Outcome: We ____ the Null Hypothesis, indicating that Lot Size is dependent on property tax value.
- Nevertheless, the relationship between Lot Size and property tax is ______, with a count of only ____ out of ____. Hence, it will not be used in the modeling phase.

Hypothesis 3: Bedrooms and Property Tax Value
- Outcome: We rejected the Null Hypothesis, indicating that the number of bedrooms are dependent on property tax value.
- However, the relationship between the number of bedrooms and property tax value is ______, with a count of ## out of #####. Thus, it will not be considered in the modeling process.


For modeling, I selected the features of _______, aiming to exceed the baseline accuracy of __%. Using Ordinary Least Squares (OLS) Linear Regression, Least Absolute Shrinkage and Selection Operator (LASSO),  Least Angle Regression (LARS),  models with a Random Seed of 123, I strived to achieve higher accuracy without over/underfitting. 

While ________ models scored higher than the baseline accuracy and exhibited consistent performance in both training and validation, ______consistently outperformed _____with an overall accuracy of ##%. The ______ regression model was chosen over the _____ Model due to the notable ##% difference between the training and validation scores in the ______ Model. The chosen  ______ Regression Model was applied to the test data.

____________ Regression performed the best of all three models with consistent ##% accuracy scores in training, validation and testing.
 
    
## Recommendations and Next Steps
Based on the findings, I propose the following recommendations and next steps:
<!-- NEEDS REWRITE Exclude the Lot Size and number of bedrooms columns during the preparation phase, as their relationship with customer property taxis shallow.
- Conduct chi-square statistical testing on the remaining columns to determine their significance in relation to property tax value, creating a subset that excludes relationships with counts less than 100 for modeling.
- Experiment with different hyperparameters to optimize the model's performance.
- Implement an exit survey for customers who have property tax valueed and consider conducting a property taxand welcome survey for new customers to gather insights on why they left their previous company for Zillow. -->

   
## Takeaways 
<!--  NEEDS REWRITE Although the models employed in this Regression project demonstrated effectiveness and accuracy, the selected features could have provided more valuable insights into the risk of property tax value. However, the data proved useful in identifying weak relationships associated with Lot Size, number of bedrooms, and even tech support in relation to property tax value. These elements can be excluded from future modeling efforts to enhance efficiency and accuracy. -->
