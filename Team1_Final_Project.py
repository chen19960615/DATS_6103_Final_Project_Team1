

### Topic: Analyzing White Wine Data
### Team 1 - Final Project
### Authors: Suhas Buravalla, Steven Chen, Siva Gogineni, Varun Shah
### Date: December 08, 2021


# We analyze White Wine Quality Dataset, sourced from https://www.sciencedirect.com/science/article/pii/S0167923609001377.
# It contains physicochemical and sensory preference data of several thousand Portuguese vinho verde white wine samples. The data was collected by CVRVV, the official certification organization of vinho verde.
# The SMART questions we pose are as follows:

# 1. Are the density, fixed acidity, residual sugar and other features collinear to each other?
# 2. Which variables among fixed acidity, residual sugar, and total sulfur dioxide are most correlated with alcohol percentage?
# 3. Which features impact the quality of the wine the most?
# 4. Are any features less significant than the other?
# 5. After checking the Confusion Matrix, F1-scores and Accuracy scores of each model, which model gives us the best results? 


# Our analyses tests four different models for predicting wine quality based on physicochemical information (Multinomial Logistic Regression, K-Nearest Neighbours, Support Vector Machines and Random Forests).

#%%

import numpy as np
import pandas as pd
import os 
import matplotlib as plt
from pandas.io.formats import style
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC 

# %%


#os.getcwd()

#%%

#os.chdir('/Users/varunshah/Documents/GitHub/DATS_6103_Final_Project_Team1')
#%%

winedata_orig = pd.read_csv('winequality-white.csv')

# %%
# We first explore our dataset 
winedata_orig.head()
#%%
winedata_orig.info()

#%%
winedata_orig.describe()

# %%
# Checking for NaN values in the dataframe 

winedata_orig.isna().sum()


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

winedata = categorize(winedata_orig, ['quality'])
#%%
winedata.info()


#%% [markdown]

# Lets understand the class, Quality of wine better and see its distribution


#%%
winedata.quality.value_counts()


#%%

# We first look at the Distribution / Histogram of Quality class

sns.set(rc={'figure.figsize':(11.7,8.27)})
X = winedata['quality']

p = sns.histplot(data = winedata, x = X, palette = 'inferno' )

p.set_title('Distribution of Wine Quality', FontSize = 20)
p.set_xlabel('Wine Quality',FontSize = 15)


#%%
#Pie Chart

sns.set(rc={'figure.figsize':(11.7,8.27)})

p = winedata.groupby('quality').size().plot(kind='pie', autopct='%1.02f%%', textprops={'fontsize': 10})

p.set_ylabel('Per Wine Quality', size=22)


#%% [markdown]

# As we can see the Wine Quality includes 7 classes, and they are not evenly distributed. 
# Quality 9 is only 0.001 percent of the total wines. Quality 3, at 0.004 percent.

#%% [markdown]

# To improve the accuracy and precision scores, we reduce the number of categories for our given data:
# Reducing the number of categories

#%%


winedata['quality'].value_counts().sort_values()

remap_quality = {3: 'Bad', 4: 'Bad', 5: 'Bad', 6: 'Average', 7: 'Good', 8: 'Good', 9: 'Good'}

winedata['quality'] = winedata['quality'].map(remap_quality).astype('category')
winedata['quality'].cat.reorder_categories(['Bad','Average','Good'], inplace=True)

winedata['quality'].value_counts()


#%%
# We see the distribution again in this reduced categories.

sns.set(rc={'figure.figsize':(11.7,8.27)})

p = winedata.groupby('quality').size().plot(kind='pie', autopct='%1.02f%%', textprops={'fontsize': 10})

p.set_ylabel('Per Wine Quality', size=22)


#%%

# We also check the correlation between all the features before we build our model.

# Correlation Plot (Including Quality as Numeric)

corr = winedata_orig.corr()
corr.style.background_gradient(cmap='viridis').set_precision(2)

#%% [markdown]

# The above shows Alcohol content having the strongest correlation with the Quality of the wine, followed by density and volatile acidity.


#%% [markdown]

# We also look at the violinplots for each feature against quality. The violin plots require the holoviews package. If not installed, see the plots on the report. 

#%%

import holoviews as hv
import hvplot.pandas

variables = [c for c in winedata.columns if 'quality' not in c]

violinlist = [ winedata.hvplot.violin(y=v, by='quality').redim.values(quality_class=['bad', 'average', 'good']) for v in variables]

violin_layout = hv.Layout(violinlist).cols(2)

hv.save(violin_layout, 'violins.png')

violin_layout



#%% [markdown]

#### VIF Check
# We also check the Multicollinearity between the independent variables to see if any of the features could be dropped if they are highly collinear.

#%%
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

#%%

winedata_X = winedata.iloc[:,:-1]
calc_vif(winedata_X)


#%%[markdown]

# We see that Density and pH level has high collinearity. We first remove these variables and test the vif score again.

# We will step by step remove the features that show high multicollinearity and keep the ones that are less than 10 which is ideal.
#%%

winedata_X = winedata.drop(columns = ['density','pH'])
winedata_X2 = winedata_X.iloc[:,:-1]
calc_vif(winedata_X2)

#%%

winedata_X2 = winedata_X.drop(columns= ['fixed acidity','total sulfur dioxide'])

winedata_X3 = winedata_X2.iloc[:,:-1]
calc_vif(winedata_X3)

#%%

winedata_X3 = winedata_X2.drop(columns=['sulphates','alcohol'])

winedata_X4 = winedata_X3.iloc[:,:-1]
calc_vif(winedata_X4)

#%% [markdown]
# The above VIF test shows us the correlation between the different independent variables. We remove them all and keep the one with VIF lower than 10.

#%%

winedata_X4.head()

#%%
# Dropping the features based on our above VIF check

winedata_vif = winedata.drop(columns = ['density',
                                       'pH',
                                       'fixed acidity',
                                       'total sulfur dioxide',
                                       'sulphates',
                                       'alcohol'])


#%%

# Pair Plot:

sns.set(rc={'figure.figsize':(11.7,8.27)})
p = sns.pairplot(data = winedata_vif,
                    hue = 'quality',
                    palette='viridis')


#%% [markdown]

# We will revisit these features as we begin building our models.



#%% [markdown]

# We now look for the best model. We will be using the below models to decide which model gives us the best results:

# 1. Multinomial Logistic Regression
# 2. K-Nearest Neighbour 
# 3. Support Vector Machine
# 4. Random Forest 

# We will split our data into Training and Test set with 4:1 ratio (75%/25%) and keeping the random state consistent across all models.



#%% [markdown]


### Multinomial Logistic Regression


#%%
#import pandas as pd 
#import numpy as np 
#import scipy as scp
import sklearn

#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report
#from sklearn import metrics 
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score

import statsmodels.api as sm


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


#%% [markdown]

# Our Logistic Regression model shows us the Train Scores at 56% along with Test Accuracy score at 56% as well, with the weighted average of Precision, Recall Scores at 56%. 
# As a model, the training and test scores are very similar and gives us the best fit. However our overall scores are very low.


# We now look at our next model.


#%% [markdown]


### K-Nearest Neighbour


#%% 


# Importing the libraries
#import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import seaborn as sb
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
#from sklearn.metrics import classification_report
#%%
#calculate_vif(dt) :
#    vif=pd.DataFrame()
#    vif['features']=dt.columns
#    vif['vif_values']=variance_inflation_factor(dt.values,i) for i in range(dt.shape[1])
#    return vif
#%%
# Importing the dataset
#dataset = pd.read_csv('C:/Users/Siva Gogineni/Documents/Python Scripts/winequality-white.csv')
dataset = pd.read_csv('winequality-white.csv')


def quality_class(i):
    if 3<=i<=5:
        return 'bad'
    elif i==6:
        return 'average'
    elif i>=7:
       return 'good'
    #else:
    #    return 'good'


#%%

# ploting heatmap
plt.figure(figsize=[19,10],facecolor='white')
sb.heatmap(dataset.corr(),annot=True)
#%%

dataset['quality_class'] = dataset['quality'].apply(quality_class)
dataset['quality_class'].value_counts()
dataset.drop(columns=['quality'],inplace=True)

#%%

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#%%

vif=pd.DataFrame()
features=dataset.iloc[:,:-1]
vif['features']=features.columns
vif['VIF_Values']=[variance_inflation_factor(features.values,i) for i in range(features.shape[1])]
print(vif)

#%%
dataset.loc[dataset['quality_class']=='average','q_number']=2
dataset.loc[dataset['quality_class']=='good','q_number']=1
dataset.loc[dataset['quality_class']=='bad','q_number']=3
pca=PCA(n_components=2)

dataset_vif_filter=dataset[['alcohol','chlorides','volatile acidity','total sulfur dioxide','alcohol','q_number']]
#dataset_vif_filter=dataset[['chlorides','sulphates','alcohol','volatile acidity','total sulfur dioxide','quality_class']]
print(dataset_vif_filter)

#%%
# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

#%%

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)
#%%

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 9, algorithm= 'brute',weights='distance',p = 2)
classifier.fit(X_train, y_train)
#%%

# Predicting a new result
print(classifier.predict(sc.transform([[7,0.27,1,20.7,0.045,70,200,1.001,6,0.45,8.8]])))
#%%


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#%%
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy score test Test \n',accuracy_score(y_test, y_pred))

print('Confusion Matrix for Test Set \n', confusion_matrix(y_test, y_pred))
print('Classification Report for Test Set \n', classification_report(y_test,  y_pred)) 




#%%

a = dataset_vif_filter.iloc[:, :-1].values
y = dataset_vif_filter.iloc[:, -1].values
pca.fit(a)
X=pca.transform(a)

#%%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

#%%

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)
#%%

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = KNeighborsClassifier(n_neighbors = 9, algorithm= 'brute',weights='distance',p = 2)
classifier.fit(X_train, y_train)
#%%

# Predicting a new result
#print(classifier.predict(sc.transform([[7,0.27,0.45]])))
#%%


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#%%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

print('Accuracy score test Test \n',accuracy_score(y_test, y_pred))

print('Confusion Matrix for Test Set \n', confusion_matrix(y_test, y_pred))
print('Classification Report for Test Set \n', classification_report(y_test,  y_pred)) 

#%%

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2  = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step =0.1),
                     np.arange(start = X_set[:, 1].min() - 10, stop = X_set[:, 1].max() + 10, step = 0.1)
                     )
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green','blue'))(i), label = j)

plt.title('K-NN (Training set)')
plt.xlabel('Alcohol')
plt.ylabel('Chlorides')
plt.legend()
plt.show()

#%%

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.5),
                     np.arange(start = X_set[:, 1].min() - 10, stop = X_set[:, 1].max() + 10, step = 0.5))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Alcohol')
plt.ylabel('Chlorides')
plt.legend()
plt.show()
#%%

plt.scatter(X[:,0],X[:,1],c=y,label=y)


#%% [markdown]

### Support Vector Machines


#%%

# We first try fitting the model with all predictors included in X except our dependent variable Quality.

winedata.head()

target = 'quality'

#%%


X_wine = winedata.loc[:, winedata.columns != target]    
y_wine = winedata.loc[:, winedata.columns == target]


#%%
X_wine.shape

y_wine.shape

#%%

# Split dataset into 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.25, random_state=8)

#%%

# Using GridSearch to find the best parameters for SVM

from sklearn.model_selection import GridSearchCV
#%%

param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train.values.ravel())

#%%
grid.best_params_

#%%
grid.best_estimator_
#%%
grid.best_score_

#%%


svc_model = SVC(C = 1000, gamma = 0.01)
#svc_model = SVC()
clf_svc_model = svc_model.fit(X_train, y_train.values.ravel())

print(clf_svc_model.score(X_test, y_test))


#%%

train_predictions = svc_model.predict(X_train)
test_predictions = svc_model.predict(X_test)

#%%

# Evaluate train-set accuracy
print('train set evaluation: ')
print('Accuracy Score \n', accuracy_score(y_train, train_predictions))

print('F1 Score: ', metrics.f1_score(y_train, train_predictions, average = 'weighted'))
print('Precision Score: \n', metrics.precision_score(y_train, train_predictions, average='weighted'))
print('Recall Score: \n', metrics.recall_score(y_train, train_predictions, average = 'weighted'))


print('Confusion Matrix for Train-set \n', confusion_matrix(y_train, train_predictions))
print('Classification Report for Train-set\n', classification_report(y_train, train_predictions))



#%% [markdown]
# The above shows the Training Set has an Accuracy Score of 95%, and weighted average of F1-scores 95%.
# We now run the test set below:

#%% 

# Evaluate Test-Set

print('test set evaluation: ')
print('Accuracy Score \n', accuracy_score(y_test, test_predictions))

print('F1 Score: ', metrics.f1_score(y_test, test_predictions, average = 'weighted'))
print('Precision Score: \n', metrics.precision_score(y_test, test_predictions, average='weighted'))
print('Recall Score: \n', metrics.recall_score(y_test, test_predictions, average = 'weighted'))

print('Confusion Matrix \n', confusion_matrix(y_test, test_predictions))
print('Classification Report \n', classification_report(y_test, test_predictions))

#%%

cm_train = confusion_matrix(y_test, test_predictions)
sns.set(font_scale=1.4) # for label size
sns.heatmap(cm_train, annot=True, annot_kws={"size": 16}) # font size

#%%

print(cross_val_score(clf_svc_model, X_test, y_test.values.ravel(), cv=3))

#%% [markdown]
# The test set shows the Accuracy score of 61%, meaning the predictions were accurate 60% of the time.
# As these classes are more evenly distributed, we also see the average of the F1-scores at 61% for the 3 classes.

#%% 
# We now try to select our features based on the correlation with Quality. 
# Starting with Alcohol which has the strongest correlation, and see if we can get a better model.

X = winedata[['alcohol',
                  'density', 'chlorides',
                  'volatile acidity','total sulfur dioxide']]
y = winedata['quality'].values


#%%
# We now repeat the same process again with these features;
# Split dataset into 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

#%%

# Using GridSearch to find the best parameters for SVM

param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)

#%%
grid.best_params_
#%%
grid.best_estimator_

#%%
grid.best_score_

#%%

svc_model_v1 = SVC(C = 1000, gamma = 1)
#svc_model = SVC(C = 10, gamma = 1)
clf_svc_model_v1 = svc_model_v1.fit(X_train, y_train)

print(clf_svc_model_v1.score(X_test, y_test))


#%%

train_predictions = svc_model_v1.predict(X_train)
test_predictions = svc_model_v1.predict(X_test)

#%%

# Evaluate train-set accuracy
print('train set evaluation: ')
print('Accuracy Score \n', accuracy_score(y_train, train_predictions))

print('F1 Score: ', metrics.f1_score(y_train, train_predictions, average = 'weighted'))
print('Precision Score: \n', metrics.precision_score(y_train, train_predictions, average='weighted'))
print('Recall Score: \n', metrics.recall_score(y_train, train_predictions, average = 'weighted'))

print('Confusion Matrix \n', confusion_matrix(y_train, train_predictions))
print('Classification Report \n', classification_report(y_train, train_predictions))



#%% [markdown]
# The above shows the Training Set has an Accuracy Score of 92%, and weighted average of F1-scores 92%.
# We now run the test set below:

#%% 

# Evaluate Test-Set

print('test set evaluation: ')
print('Accuracy Score \n', accuracy_score(y_test, test_predictions))

print('F1 Score: ', metrics.f1_score(y_test, test_predictions, average = 'weighted'))
print('Precision Score: \n', metrics.precision_score(y_test, test_predictions, average='weighted'))
print('Recall Score: \n', metrics.recall_score(y_test, test_predictions, average = 'weighted'))


print('Confusion Matrix \n', confusion_matrix(y_test, test_predictions))
print('Classification Report \n', classification_report(y_test, test_predictions))


#%%
cm_train = confusion_matrix(y_test, test_predictions)
sns.set(font_scale=1.4) # for label size
sns.heatmap(cm_train, annot=True, annot_kws={"size": 16}) # font size

#%%

print(cross_val_score(clf_svc_model_v1, X_test, y_test, cv=5))

#%% [markdown]
# The test set shows the Accuracy score of 60%, meaning the predictions were accurate 60% of the time.
# As these classes are more evenly distributed, we also see the average of the F1-scores at 60% for the 3 classes.
# We notice that keeping only 5 features vs 11 features, has only made a difference of 0.07% in the scores.




#%% [markdown]

# We now select features based on the VIF check, the features with multicolinearity less than 10 is accepted.
# This includes Volatile Acidity, Citric Acid, Residual Sugar, Chlorides and Free Sulphur Dioxide.

#%%

X = winedata[['volatile acidity',
                  'citric acid', 'residual sugar',
                  'chlorides','free sulfur dioxide']]
y = winedata['quality']


#%%
X.columns
X.shape

y.name
y.shape

#%%
# Split dataset into 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

#%%

param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train.values.ravel())

#%%
grid.best_params_
#%%
grid.best_estimator_

#%%
grid.best_score_

#%%

svc_model_vif = SVC(C = 1000, gamma = 1)
#svc_model_vif = SVC()
clf_svc_model_vif = svc_model_vif.fit(X_train, y_train.values.ravel())

print(clf_svc_model_vif.score(X_test, y_test))


#%%

train_predictions = svc_model_vif.predict(X_train)
test_predictions = svc_model_vif.predict(X_test)

#%%

# Evaluate train-set accuracy
print('train set evaluation: ')
print('Accuracy Score \n', accuracy_score(y_train, train_predictions))

print('F1 Score: ', metrics.f1_score(y_train, train_predictions, average = 'weighted'))
print('Precision Score: \n', metrics.precision_score(y_train, train_predictions, average='weighted'))
print('Recall Score: \n', metrics.recall_score(y_train, train_predictions, average = 'weighted'))

print('Confusion Matrix \n', confusion_matrix(y_train, train_predictions))
print('Classification Report \n', classification_report(y_train, train_predictions))



#%% [markdown]
# The above shows the Training Set has an Accuracy Score of 95%, and weighted average of F1-scores 96%.
# We now run the test set below:

#%% 

# Evaluate Test-Set

print('test set evaluation: ')
print('Accuracy Score \n', accuracy_score(y_test, test_predictions))

print('F1 Score: ', metrics.f1_score(y_test, test_predictions, average = 'weighted'))
print('Precision Score: \n', metrics.precision_score(y_test, test_predictions, average='weighted'))
print('Recall Score: \n', metrics.recall_score(y_test, test_predictions, average = 'weighted'))

print('Confusion Matrix \n', confusion_matrix(y_test, test_predictions))
print('Classification Report \n', classification_report(y_test, test_predictions))



#%% [markdown]

# The above feature selection based on VIF check, brings the accuracy score and the average F1-score to 58%.

# Looking at all the models tested, we think the model with features from highest correlation is our best model.
# This includes predictors: 'alcohol','density', 'chlorides','volatile acidity','total sulfur dioxide'

#%% [markdown]

### Random Forest

#%%

# Using all features gives best performance at ~70%.
X_train, X_test, y_train, y_test = train_test_split(winedata[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']], winedata['quality'], random_state=8)

# Using top 5 important features given in feature importance plot below decreases performance about 1.5%
# X_train, X_test, y_train, y_test = train_test_split(winedata[['alcohol', 'density', 'volatile acidity', 'free sulfur dioxide', 'total sulfur dioxide']], winedata['quality'], random_state=8)

# Using top 5 features correlated with quality decreases performance about 1.5%
# X_train, X_test, y_train, y_test = train_test_split(winedata[['alcohol', 'density', 'chlorides', 'volatile acidity','total sulfur dioxide']], winedata['quality'], random_state=8)


#%%
from sklearn.ensemble import RandomForestClassifier

opts = {'bootstrap': True, 'class_weight': None, 'criterion': 'gini',
        'max_depth': None, 'max_features': None, 'max_leaf_nodes': None,
        'min_samples_leaf': 2, 'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.0, 'max_samples': None, 'max_features': 'auto', 'n_jobs': -1,
        'verbose': 0, 'warm_start': False,
        'n_estimators': 300, 'random_state': None}

rf = RandomForestClassifier(**opts)  # instantiate
rf.fit(X_train, y_train)

print('Random Forest model accuracy (with the test set):', rf.score(X_test, y_test))
print('Random Forest model accuracy (with the train set):', rf.score(X_train, y_train))
print('Classification Report on test set\n', classification_report(y_test, rf.predict(X_test)))
print('Classification Report on training set\n', classification_report(y_train, rf.predict(X_train)))
print('Confusion Matrix on test set\n', confusion_matrix(y_test, rf.predict(X_test)))


# Feature Importance
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

importances = pd.Series(importances, index=X_train.columns)

plt.figure(figsize=(8, 6))
importances.plot.bar(yerr=std)
plt.title("Feature importances using MDI")
plt.xticks(rotation = 30)
plt.ylabel("Mean decrease in impurity")
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('rf_features.png')
plt.show()
plt.close()

#%%
# decision tree vizualisation
estimator = rf.estimators_[1]
from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = X_train.columns,
                class_names = y_train.unique(),
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
# from graphviz import render
# render('dot', 'png', 'tree.dot') 

#%% [markdown]

### Looking at all the models, we think KNN and Random Forest gave us the best results with the below Accuracy Scores and F1- Scores:


#%% [markdown]

### Conclusion:

# Our analysis, answer majority of the questions we posed at the beginning.

# 1. Are the density, fixed acidity, residual sugar and other features collinear to each other?
# The VIF Check showed us volatile acidity, citric acid, residual sugar, chlorides and free sulphur dioxide having the least collinearity between each other. 
# This shows us that changing either of these features does not impact the characteristics of other features.


# 2. What is the correlation between all the features with alcohol?
# The correlation matrix showed us Density, Residual Sugar, Total Sulphur Dioxide are highly correlated with alcohol and has negative effect on the alcohol content.
# For a wine to reduce its alcohol level, they could make changes to any of these features to see the difference.

# 3. Which features impact the quality of the wine the most?
# We again look at the correlation matrix and the scores from our models, and can state that Alcohol, Density, Chlorides, Volatile Acidity & Total Sulphur Dioxide impacts Quality the most statistically.

# 4. Are any features less significant than the other?
# After testing different models, we found pH level and sulphates as the least significant.

# 5. Out of all the different models, which model gives us the best results?
# Comparing all the models, Random Forest and K-NN gives us the best results.


