# Loan-Default-Prediction

# Description:
Built a predicitve modelt to predict the probabilty and determine if that customer would default on loan, so the bank could identify the applications to approve loan and reduce their NPA(Non-Performing Assets). The Dataset had 5,32,500 records and 45 variables. 

# Approach:
1) Cleanned variables and converted to the required format
2) Dropped categorical vairables which added no value to prediction
3) Dropped vairables with more than 60% missing values
4) Checked for correlation and skewness
5) Performed feature engineering 
6) Identified important variables using random forest
7) One hot encoded categorical variables
8) Built a Gradient Boosting(XGBoost) model to predict probabilty of defaults and optimzed the model using cross validation
