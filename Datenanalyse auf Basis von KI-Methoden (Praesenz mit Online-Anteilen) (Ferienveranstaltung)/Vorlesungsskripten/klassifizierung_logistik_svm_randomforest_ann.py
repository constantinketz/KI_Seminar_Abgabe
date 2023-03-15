#!/usr/bin/env python
# coding: utf-8

# ## Datensatz
# Datensatz für Vorhersagen das Überleben von Patienten mit Herzinsuffizienz anhand von medizinische Merkmale wie Serumkreatinin, Ejektionsfraktion, ....

# ### Import von Bibliotheken

# In[10]:


import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report

import warnings
warnings.filterwarnings('ignore')


# ### Import der Datensatz

# In[4]:



data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
df = data.copy()

data.head(10)


# In[5]:


df.describe()


# ### Data Visualization

# In[6]:


f, ax = plt.subplots(figsize=(14,14))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".1f", ax=ax)
plt.show()


# ## Model Construction

# In[7]:


inp_data = data.drop(data[['DEATH_EVENT']], axis=1)
out_data = data[['DEATH_EVENT']]

scaler = StandardScaler()
inp_data = scaler.fit_transform(inp_data)

X_train, X_test, y_train, y_test = train_test_split(inp_data, out_data, test_size=0.2, random_state=42)


# In[8]:


print("X_train Shape : ", X_train.shape)
print("X_test Shape  : ", X_test.shape)
print("y_train Shape : ", y_train.shape)
print("y_test Shape  : ", y_test.shape)


# In[11]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()


# In[12]:


cf_matrix = confusion_matrix(y_pred.T, y_test)
sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")


# **Support Vector Machine Algorithm**
# <br>
# <img src="https://cdn-images-1.medium.com/max/1600/1*TudH6YvvH7-h5ZyF2dJV2w.jpeg" width="500px"/><br>
# <img src="https://aitrends.com/wp-content/uploads/2018/01/1-19SVM-2.jpg" width="500px"/>

# In[13]:


from sklearn.svm import SVC
clf = SVC() 
clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n",classification_report(y_pred, y_test))


# In[14]:


# find best parameters with SVC | Step 1
kernels = list(['linear', 'rbf', 'poly', 'sigmoid'])
c = list([0.01, 0.1, 1])
clf = SVC()
clf.fit(X_train, y_train) 
param_grid = dict(kernel=kernels, C=c)
grid = GridSearchCV(clf, param_grid, cv=10, n_jobs=-1)
grid.fit(X_train, y_train)
grid.best_params_


# In[15]:


clf = SVC(C=0.1, kernel='linear') 
clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n",classification_report(y_pred, y_test))


# Random Forest ist ein Gemeinschaftsmodell, bei dem mehrere Entscheidungsbäume kombiniert werden, um ein stärkeres Modell zu erhalten. Das daraus abgeleitete Modell ist robuster und genauer und kann mit Überanpassungen besser umgehen als konstitutive Modelle.
# 
# ## Grundlegende Theorie
# Random Forest besteht aus einer Reihe von Entscheidungsbäumen, die mit der "Bagging-Methode" kombiniert werden, um Klassifizierungs- und Regressionsergebnisse zu erhalten. Bei der Klassifizierung wird die Ausgabe anhand der Mehrheitsentscheidung berechnet, während bei der Regression der Durchschnitt berechnet wird.
# 
# Random Forest erstellt ein robustes und genaues Modell, das eine Vielzahl von Eingabedaten mit binären, kategorialen und kontinuierlichen Merkmalen verarbeiten kann.
# 
# ![](https://miro.medium.com/max/592/1*i0o8mjFfCn-uD79-F1Cqkw.png)
# 

# In[16]:


# Import the necessary packages
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

clf = RandomForestClassifier(random_state=0)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
    
print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('Random Forest Classifier f1-score      : {:.4f}'.format(f1_score( y_test , y_pred)))
print('Random Forest Classifier precision     : {:.4f}'.format(precision_score(y_test, y_pred)))
print('Random Forest Classifier recall        : {:.4f}'.format(recall_score(y_test, y_pred)))
print("Random Forest Classifier roc auc score : {:.4f}".format(roc_auc_score(y_test,y_pred)))
print("\n",classification_report(y_pred, y_test))
    
plt.figure(figsize=(6,6))
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")
plt.title("RandomForestClassifier Confusion Matrix (Rate)")
plt.show()
    
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=["FALSE","TRUE"],
                yticklabels=["FALSE","TRUE"],
                cbar=False)
plt.title("RandomForestClassifier Confusion Matrix (Number)")
plt.show()


# In[41]:


param_grid = {
    "n_estimators": [100, 500, 1000],
    "max_features": [0.5,1,'auto'],
    "max_depth": [1,2,3,4,None],
    "min_samples_split": [2,5,8]
}

clf = RandomForestClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_


# In[ ]:


clf = RandomForestClassifier(
    n_estimators=1000,
    max_features=0.5,
    max_depth=3,
    min_samples_split=5,
    random_state=0
)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('Random Forest Classifier f1-score      : {:.4f}'.format(f1_score( y_test , y_pred)))
print('Random Forest Classifier precision     : {:.4f}'.format(precision_score(y_test, y_pred)))
print('Random Forest Classifier recall        : {:.4f}'.format(recall_score(y_test, y_pred)))
print("Random Forest Classifier roc auc score : {:.4f}".format(roc_auc_score(y_test,y_pred)))
print("\n",classification_report(y_pred, y_test))
    
plt.figure(figsize=(6,6))
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")
plt.title("RandomForestClassifier Confusion Matrix (Rate)")
plt.show()
    
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=["FALSE","TRUE"],
                yticklabels=["FALSE","TRUE"],
                cbar=False)
plt.title("RandomForestClassifier Confusion Matrix (Number)")
plt.show()


# ## Deep Learning
# 

# In[23]:



from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras import callbacks
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

model = Sequential()

# layers
model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
history = model.fit(X_train, y_train, batch_size = 16, epochs = 80, validation_split=0.25)



val_accuracy = np.mean(history.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_accuracy is', val_accuracy*100))

# Predicting the test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.4)
np.set_printoptions()


print(classification_report(y_test, y_pred))

# Getting the confusion matrix
cmap1 = sns.diverging_palette(2, 165, s=80, l=55, n=9)
plt.subplots(figsize=(10,7))
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':25})




# In[ ]:




