"""
Gruppenabgabe von
    Malte Neumann,
    Alexandra Weigel,
    Lukas Kleinert,
    Constantin Ketz,
    Moritz Kenk,
    Romy Glück,
    Daniel Junginger,
    Timo Rahel
"""
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as holdout
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('StudentsPerformance.csv')
df.head()
df.shape
df.describe()
df.dtypes
df.isnull().sum()
ax = sns.lmplot(x='reading score', y='math score', data=df, palette='Set1')
ax = sns.lmplot(x='writing score', y='math score', data=df, palette='Set2')

# Umwandlung von Objektbeschriftungen in kategorische
categories = ['gender', 'race/ethnicity',
              'parental level of education',
              'lunch', 'test preparation course']
df[categories] = df[categories].astype('category')
df.dtypes
# df.head()


# Namen umformatieren
df.rename(columns={'race/ethnicity': 'raceethnicity',
                   'parental level of education': 'parentaleducation',
                   'test preparation course': 'pretest'}, inplace=True)

# Umwandlung von Kategoriebezeichnungen in numerische Werte mit LabelEncoder
label = LabelEncoder()
label.fit(df.gender.drop_duplicates())
df.gender = label.transform(df.gender)
label.fit(df.raceethnicity.drop_duplicates())
df.raceethnicity = label.transform(df.raceethnicity)
label.fit(df.parentaleducation.drop_duplicates())
df.parentaleducation = label.transform(df.parentaleducation)
label.fit(df.lunch.drop_duplicates())
df.lunch = label.transform(df.lunch)
label.fit(df.pretest.drop_duplicates())
df.pretest = label.transform(df.pretest)
df.dtypes
df.head()

f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = sns.heatmap(df.corr(), annot=True, cmap='cool')


def cross_val(model):
    pred = cross_val_score(model, x, y, cv=10)
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


x = df.drop(['math score'], axis=1)
y = df['math score']
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)


############
# Linear Regression with p-values
x_train = sm.add_constant(x_train)
est = sm.OLS(y_train, x_train)
est2 = est.fit()
print(est2.summary())


# Predicting the charges
x_test = sm.add_constant(x_test)
test_pred = est2.predict(x_test)
train_pred = est2.predict(x_train)
# Comparing the actual output values with the predicted values
sns.scatterplot(y_test)
sns.scatterplot(test_pred)

df = pd.DataFrame({'Actual': y_test, 'Predicted': test_pred})
df.head()


print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

### LASSO REGRESSION####

lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,
                        0.3, 0.6, 1],
                max_iter=50000, cv=10)
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


pd.Series(model.coef_, x_train.columns).sort_values(ascending=True).plot(kind="bar")
coefs_lasso = pd.Series(model.coef_, index=x_train.columns)
print(coefs_lasso.head(10))

###### RIDGE REGRESSION####

ridge = RidgeCV(alphas=[0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
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
pd.Series(model.coef_, x_train.columns).sort_values(ascending=True).plot(kind="bar")
coefs_ridge = pd.Series(model.coef_, index=x_train.columns)
print(coefs_ridge.head(10))
