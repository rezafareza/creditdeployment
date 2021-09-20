# credit_risk

#
Credit risk modeling is used to predict the ability of borrower to pay their credit.

Dataset used is 'credit_risk_dataset' that contains 20 columns (features) and 32580 rows.
# Data preprocessing
Data is divided into categorical and numerical variables.
## Handling missing values
To handle missing values in numerical, we use the median value of the column. We first check the statistics of the dataset to see the distribution of the data and also to find outliers.
## Handling categorical variables
OneHotEnconding is used to transform categorical variable into boolean values and expand the columns.
## Outliers handling
By looking at the data statistics, there are 3 variables that have outliers, namely Person's age (person_age), Person's income (person_income), and Person's employement duration (person_emp_length). We simply drop the values by excluding data that is higher than 3/4 quartile.
