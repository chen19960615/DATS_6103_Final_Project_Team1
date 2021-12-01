#%%
import pandas as pd 
import numpy as np 
import scipy as scp
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import statsmodels.api as sm
import matplotlib.pyplot as plt
#%%
wine = pd.read_csv("winequality-white.csv")
#%%
#wine['quality'] = wine['quality'].astype(object)
# %%

# To change Type, 'Quality' to Category type
def categorize(dframe, cat_list):
    dframe = dframe.copy(deep=True)
    for col_name in dframe.columns :
        if col_name in cat_list:
           dframe[col_name] = dframe[col_name].astype('category') 
        else:
            pass
        
    return dframe

#%%

winedata = categorize(wine, ['quality'])
#%%
winedata.info()

#%%
winedata.quality.value_counts()
winedata.quality.value_counts().sum()



#%%

# To improve the accuracy and precision scores, we reduce the number of categories for our given data:
# Reducing the number of categories

winedata['quality'].value_counts().sort_values()

remap_quality = {3: 'bad', 4: 'bad', 5: 'bad', 6: 'average', 7: 'good', 8: 'good', 9: 'good'}

winedata['quality'] = winedata['quality'].map(remap_quality).astype('category')

winedata['quality'].value_counts()
# %%
# initial logit model with all variables 
#x = winedata.drop('quality', axis=1)
#y = winedata['quality']
#x = winedata[['volatile acidity',
                  #'citric acid', 'residual sugar',
                  #'chlorides','free sulfur dioxide']]
#y = winedata['quality']

x= winedata[['volatile acidity', 'chlorides', 'density', 'alcohol', 'total sulfur dioxide']]
y= winedata['quality']
#%%

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.25, random_state = 8)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



# %%
model1 = LogisticRegression(random_state=8, multi_class="multinomial", penalty="none", solver="newton-cg", max_iter = 5000).fit(x_train, y_train)
test_preds = model1.predict(x_test)
train_preds = model1.predict(x_train)

#%%
params = model1.get_params()
print(params)

#%%
print('Intercept: \n', model1.intercept_)
print('Coefficients: \n', model1.coef_)

#%%
import numpy as np
np.exp(model1.coef_)


#%%
#Use statsmodels to assess variables

logit_model=sm.MNLogit(y_train,sm.add_constant(x_train))
logit_model
result=logit_model.fit()
stats1=result.summary()
stats2=result.summary2()
print(stats1)
print(stats2)

# %%


# Evaluate train-set accuracy
print('train set evaluation: ')
print('Accuracy Score \n', accuracy_score(y_train, train_preds))

print('F1 Score: ', metrics.f1_score(y_train, train_preds, average = 'weighted'))
print('Precision Score: \n', metrics.precision_score(y_train, train_preds, average='weighted'))
print('Recall Score: \n', metrics.recall_score(y_train, train_preds, average = 'weighted'))


cm_train = confusion_matrix(y_train, train_preds)
print('Confusion Matrix \n', confusion_matrix(y_train, train_preds))
print('Classification Report \n', classification_report(y_train,train_preds))


#%%
# Evaluate test-set accuracy
print('test set evaluation: ')
print('Accuracy Score \n', accuracy_score(y_test, test_preds))

print('Precision Score: \n', metrics.precision_score(y_test, test_preds, average='weighted'))
print('Recall Score: \n', metrics.recall_score(y_test, test_preds, average = 'weighted'))
print('Confusion Matrix \n', confusion_matrix(y_test, test_preds))

print('Classification Report \n', classification_report(y_test, test_preds))

# %%
#Logit model with feature selection 1 

