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

import statsmodels.api as sm
import matplotlib.pyplot as plt
#%%
wine = pd.read_csv("winequality-white.csv")
# %%
# initial logit model with all variables 
x = wine.drop('quality', axis=1)
y = wine['quality']
#%%
wine['quality'] = wine['quality'].astype(object)
#%%

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state = 5)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



# %%
model1 = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(x_train, y_train)
preds = model1.predict(x_test)

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
