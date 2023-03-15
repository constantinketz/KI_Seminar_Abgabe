#!/usr/bin/env python
# coding: utf-8

# <font size=5 > <p style="color:purple"> Welche Faktoren beeinflussen den Preis von Krankenversicherungen?

# <font size='2'>Viele Faktoren, die sich auf die Höhe Ihrer Krankenversicherungsbeiträge auswirken, liegen nicht in Ihrem Einflussbereich. Trotzdem ist es gut, wenn Sie wissen, welche das sind. Hier sind einige Faktoren, die sich auf die Höhe der Krankenversicherungsprämien auswirken
# 
# * **Alter:** Alter des Hauptbegünstigten
#

# * **Geschlecht:** Geschlecht des Versicherungsvertragsnehmers, weiblich, männlich
# 
# * **bmi:** Body-Mass-Index, bietet ein Verständnis des Körpers, Gewichte, die relativ hoch oder niedrig im Verhältnis zur Höhe, objektive Index des Körpergewichts (kg / m ^ 2) mit dem Verhältnis von Höhe zu Gewicht, idealerweise 18,5 bis 24,9 sind
# 
# * **Kinder:** Anzahl der krankenversicherten Kinder / Anzahl der unterhaltsberechtigten Personen
# 
# * **Raucher:** Rauchen
# 
# * **Region:** Wohngebiet des Begünstigten in den USA, Nordosten, Südosten, Südwesten, Nordwesten
# 
# 

# <font size=5><p style="color:purple"> EDA and Visualizations 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('insurance.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# <font size='2' font>Analysieren wir nun die Versicherungskosten nach Alter, BMI und Kindern nach dem Faktor Rauchen 

# In[8]:


ax = sns.lmplot(x = 'age', y = 'charges', data=df, palette='Set1')
ax = sns.lmplot(x = 'bmi', y = 'charges', data=df, palette='Set2')


# In[9]:


##Umwandlung von Objektbeschriftungen in kategorische
df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype('category')
df.dtypes
#df.head()


# In[10]:


##Umwandlung von Kategoriebezeichnungen in numerische Werte mit LabelEncoder
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
label.fit(df.sex.drop_duplicates())
df.sex = label.transform(df.sex)
label.fit(df.smoker.drop_duplicates())
df.smoker = label.transform(df.smoker)
label.fit(df.region.drop_duplicates())
df.region = label.transform(df.region)
df.dtypes
df.head()


# In[11]:


f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = sns.heatmap(df.corr(), annot=True, cmap='cool')


# <font size='2' font>Keine Korrelation, außer mit dem Rauchen 

# <font size=5><p style="color:purple"> Lineare Regression

# <font size=2><p style="color:blue"> Funktionen für Evaluation der Ergebnisse

# In[12]:


from sklearn import metrics
from sklearn.model_selection import cross_val_score


def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def regression_analysis(X, y, model):
    
    is_statsmodels = False
    is_sklearn = False
    
    # check for accepted linear models
    if type(model) in [sklearn.linear_model._base.LinearRegression,
                       sklearn.linear_model._ridge.Ridge,
                       sklearn.linear_model._ridge.RidgeCV,
                       sklearn.linear_model._coordinate_descent.Lasso,
                       sklearn.linear_model._coordinate_descent.LassoCV,
                       sklearn.linear_model._coordinate_descent.ElasticNet,
                       sklearn.linear_model._coordinate_descent.ElasticNetCV,
                      ]:
        is_sklearn = True
    elif type(model) in [statsmodels.regression.linear_model.OLS, 
                         statsmodels.base.elastic_net.RegularizedResults,
                        ]:
        is_statsmodels = True
    else:
        print("Only linear models are supported!")
        return None
    
    
    
    has_intercept = False
    
    if is_statsmodels and all(np.array(X)[:,0]==1):
        # statsmodels add_constant has been used already
        has_intercept = True  
    elif is_sklearn and model.intercept_:
        has_intercept = True
        

    
    if is_statsmodels:
        # add_constant has been used already
        x = X
        model_params = model.params
    else: # sklearn model
        if has_intercept:
            x = sm.add_constant(X)
            model_params = np.hstack([np.array([model.intercept_]), model.coef_])
        else:
            x = X
            model_params = model.coef_
        
    #y = np.array(y).ravel()
    
    # define the OLS model
    olsModel = sm.OLS(y, x)
    
    pinv_wexog,_ = pinv_extended(x)
    normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))
    
    
    return sm.regression.linear_model.OLSResults(olsModel, model_params, normalized_cov_params)
    


# In[13]:


from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy import stats

x = df.drop(['charges'], axis = 1)
y = df['charges']
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)


############
####Linear Regression with p-values
import statsmodels.api as sm
from scipy import stats
x_train=sm.add_constant(x_train)
est = sm.OLS(y_train, x_train)
est2 = est.fit()
print(est2.summary())



##Predicting the charges
x_test = sm.add_constant(x_test)
test_pred = est2.predict(x_test)
train_pred = est2.predict(x_train)
##Comparing the actual output values with the predicted values
sns.scatterplot(y_test,test_pred)

df = pd.DataFrame({'Actual': y_test, 'Predicted': test_pred})
df.head()


print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)


# <font size=5><p style="color:purple"> Lasso Regression

# In[14]:


from sklearn.linear_model import Lasso, LassoCV

lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(x_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)


model = Lasso(alpha=alpha, 
              precompute=True, 
#               warm_start=True, 
              positive=True, 
              selection='random',
              random_state=42)
model.fit(x_train, y_train)

test_pred = model.predict(x_test)
train_pred = model.predict(x_train)



print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)


pd.Series(model.coef_, x_train.columns).sort_values(ascending = True).plot(kind = "bar")
coefs_lasso = pd.Series(model.coef_, index = x_train.columns)
print(coefs_lasso.head(10))


# <font size=5><p style="color:purple"> Ridge Regression

# In[15]:


from sklearn.linear_model import Ridge, RidgeCV
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(x_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

model = Ridge(alpha=alpha, solver='cholesky', tol=0.0001, random_state=42)
model.fit(x_train, y_train)
pred = model.predict(x_test)

test_pred = model.predict(x_test)
train_pred = model.predict(x_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# Plot important coefficients
pd.Series(model.coef_, x_train.columns).sort_values(ascending = True).plot(kind = "bar")
coefs_ridge = pd.Series(model.coef_, index = x_train.columns)
print(coefs_ridge.head(10))


# <font size=5><p style="color:purple"> Deep Learning

# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()

model.add(Dense(x_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer=Adam(0.01), loss='mse')

r = model.fit(x_train, y_train,
              validation_data=(x_test,y_test),
              batch_size=1,
              epochs=40)

test_pred = model.predict(x_test)
train_pred = model.predict(x_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)





# In[ ]:




