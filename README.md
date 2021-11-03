# DATS_6103_Final_Project_Team1

Possible datasets:

https://data.galaxyzoo.org/

https://www.kaggle.com/kashnitsky/topic-1-exploratory-data-analysis-with-pandas/data?select=adult_train.csv

US Census Dataset:

Using the US Census from 1994 (https://cseweb.ucsd.edu/classes/sp15/cse190-c/reports/sp15/048.pdf), explore relationships between several demographic variables, including sex, race, education, etc. and income, where income is a categorical variable (> or < $50,000 per year). 

SMART Questions:

Which variable, race, sex, or education is a bigger predictor of higher income? 

How does education level correlate with the work classes? 

What work classes and incomes correlate with the various non-US nationalities? 


The data is a mix of categorical and continuous variables, so data exploration shall consist of a combination of histograms, violin plots, and other plots. Modeling of income (> or < $50,000 per year) will be done mainly through linear regression on continuous variables (age and education) for each set of categorical variables (race, sex, workclass)???


White Wine Quality Dataset:

This dataset was produced by Cortez et al. in Modeling wine preferences by data mining from physicochemical properties (https://www.sciencedirect.com/science/article/pii/S0167923609001377). It contains physicochemical and sensory preference data of several thousand Portuguese vinho verde white wine samples. The data was collected by CVRVV, the official certification organization of vinho verde. 

SMART Questions:


Q1- Are fixed acidity, volatile acidity, residual sugar, density, citric acid, sulphates, chlorides, sulphur dioxide, alcohol and pH independent of each other?If not which combinations of those variables are independent of each other?

Q2- Which variables among fixed acidity, residual sugar, and total sulfur dioxide are most correlated with alcohol percentage?

Q3- Is residual sugars, or another chemical concentration (citric acid, Chlorides, Total sulfur dioxide, Sulphates, alcohol) the biggest predictor of subjective quality?



The paper tested three models for predicting wine quality based on physicochemical information (SVMs, multiple regression, Neural networks). We can test these models, as well as other models, such as random forests, and K-nearest neighbors. 
