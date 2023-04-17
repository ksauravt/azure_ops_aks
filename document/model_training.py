#!/usr/bin/env python
# coding: utf-8

# ## Loan Default Classification

# <b>Machine learning in the FinTech industry</b><br>
# <b>Problem statement:</b> The company is facing challenges in the loan procedure, specifically the issue of defaulters. The company wants to transform its loan procedure using Machine Learning model to improve decision-making and prevent defaulters.<br>
# <b>Solution:</b> The company can use AI and automation to analyze customer data in real-time and predict the likelihood of a customer defaulting on their loan. Machine learning models such as logistic regression, decision trees, random forests, and neural networks can be used to make better decisions regarding loan approval. The data analytics team should consider relevant parameters such as having enough data to train the model accurately, choosing the right features to predict the likelihood of default, and considering the performance metrics to evaluate the model. The future of AI and automation in the loan procedure looks promising as it can improve decision-making, reduce the risk of defaulters, and provide better customer service. This can have a positive impact on the economy by improving the financial health of individuals and businesses, leading to increased investment and economic growth.
# 

# <b>how can our company transform with the use of AI and automation to solve the loan procedure and prevent defaulters?</b><br>
# Overview of how we could implement this:
# With the help of AI and automation, we can analyze customer data in real-time and predict the likelihood of a customer defaulting on their loan. We can use machine learning models such as logistic regression, decision trees, random forests, and neural networks to make better decisions regarding loan approval.<br>
# <b>What parameters should we consider when building these machine learning models?</b><br>
# Data analytics team should consider several parameters when building these models. Firstly, we need to ensure that we have enough relevant data to train the model accurately. Secondly, we need to choose the right features to predict the likelihood of default. And finally, we need to consider the performance metrics such as precision, recall, and accuracy that we're using to evaluate the model.<br>
# <b>What do you think the future of AI and automation in the loan procedure looks like?</b><br>
# I believe the future looks very promising for AI and automation in the loan procedure. As we continue to collect more data and improve our machine learning models, we'll be able to make better decisions regarding loan approval and reduce the risk of defaulters. Additionally, we can provide better customer service through automation and personalized loan servicing, which could lead to higher customer satisfaction and loyalty.<br>
# <b>How do you think this will affect the recession in the market?</b><br>
# I think this will have a positive impact on the economy. By reducing the risk of defaulters, the financial health of individuals and businesses will improve, which could lead to increased investment and economic growth. It's a win-win situation for both our company and the economy.<br>
# 

# <b>MLOps</b><br>
# The MLOps pipeline can involve data collection and preprocessing, feature engineering, model training, model deployment, and monitoring. The pipeline can be automated using tools such as docker, Azure ML and Kubernetes. The pipeline can be deployed on cloud infrastructure such as AWS, GCP, or Azure.
# 

# <b>About Dataset</b><br>
# A simulated financial dataset has been generated using genuine information from a financial organization. The dataset has been altered to eliminate any identifying characteristics and the figures have been altered to prevent any linkage to the original source (the financial institution). The purpose of using this dataset is to give trainee a simple financial dataset to use when practicing financial analytics for POC.
# 

# <b>Highlights of the Loan Default Classification: </b><br>
#       -Classification, Imbalanced Data, and PR Curve<br>
# <b>Contents</b><br>
# 1.	EDA <br>
#     Check NaN values<br>
#     Data overview<br>
#     Feature engineering<br>
#     Data distribution<br>
# 
# 2.	Modeling<br>
#     Train test split<br>
#     Standardization<br>
#     Upsampling by SMOTE<br>
#     Logistic regression<br>
#     Support vector machine<br>
#     Random forest<br>
#     LightGBM<br>
#     XGBoost<br>
#     Model assessment<br>
#     ROC curve<br>
#     PR curve<br>
# 3.	Conclusion<br>
# 

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
import os
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import seaborn as sns


# In[2]:


dataset = pd.read_csv('data/FinTech_Dataset.csv',index_col=0)
print(dataset.head())


# In[3]:


dataset.shape

# EDA
# Check NaN values
# In[4]:


dataset.isna().sum()
#As there are no NaN values in this data, invalid values are not a major concern.


# ## Data overview

# In[5]:


dataset.describe()


# The column labeled "Employed" is of categorical type, while the "Bank Balance" and "Annual Salary" columns are numerical. 
# Our objective is to perform a binary classification task based on the target column "Defaulted."

# ## Feature engineering

# In[6]:


dataset.insert(3, 'Saving Rate', dataset['Bank Balance'] / dataset['Annual Salary'])
print(dataset.head())


# We generate a new feature named "Saving Rate" based on the "Bank Balance" and "Annual Salary" data. The Saving Rate feature provides insight into the spending habits of each user. Generally, a user with a higher Saving Rate is considered less likely to default. We will investigate the relationship between these variables in greater detail later on.

# ## Data distribution

# Default distribution

# In[7]:


tbl = dataset['Defaulted?'].value_counts().reset_index()
tbl.columns = ['Status', 'Number']
tbl['Status'] = tbl['Status'].map({1 :'Defaulted', 0 :'Not defaulted'})
print(tbl)


# In[8]:


fig = px.pie(tbl,
             values='Number', 
             names = 'Status',
             title='Default Status')
fig.show() 


# Loan defaults would only impact 3% of customers, creating in an imbalanced classification.

# Employed distribution

# In[9]:


tbl = dataset['Employed'].value_counts().reset_index()
tbl.columns = ['Status', 'Number']
tbl['Status'] = tbl['Status'].map({1 :'Employed', 0 :'Unemployed'})
tbl


# In[10]:


fig = px.pie(tbl,
             values='Number', 
             names = 'Status',
             title='Employed Status')
fig.show()


# In[11]:


tbl = dataset.copy()
tbl['Employed'] = tbl['Employed'].replace({1 :'Employed', 0 :'Unemployed'})
tbl['Defaulted?'] = tbl['Defaulted?'].replace({1 :'Defaulted', 0 :'Not defaulted'})


# In[12]:


fig = px.sunburst(tbl, 
                  path=['Employed','Defaulted?'],
                  title='Relationship between Employment and Loan Default')
fig.show()


# Contingency table

# In[13]:


tbl = pd.crosstab(dataset['Employed'],dataset['Defaulted?'])
print(tbl)


# Pearson’s  χ2  test for independence

# In[14]:


chi2, p, dof, ex = chi2_contingency(tbl)
print("p-value:", p)


# Conclusion:
# As their p-value is between 0.0005 and 0.05, we draw the conclusion that they are not independent. Employed status can therefore be used to predict default.

# Bank Balance distribution

# In[15]:


fig = px.histogram(dataset, x="Bank Balance", color='Defaulted?', 
                   marginal="box", # or violin, rug
                   hover_data=dataset.columns)
fig.show()


# We find that this is an asymmetric distribution, with many people having zero bank balance.
# 
# Let's further check this by calculating number of accounts with less than 10 dollars.

# In[16]:


(dataset['Bank Balance'] <= 10).sum()


# Conclusion:
# 
# Approximately 500 individuals have hardly saved any money in their bank accounts, which could pose a risk for loan defaults. 
# Surprisingly, those who have defaulted on their loans tend to have a higher balance in their bank accounts. 
# This observation may seem counterintuitive and suggests the presence of confounding factors. 
# It is possible that individuals with a higher bank balance may have easier access to loans, leading to a higher number of defaults.

# Annual Salary distribution

# In[17]:


fig = px.histogram(dataset, x="Annual Salary",
                   color="Defaulted?",
                   marginal="box", # or violin, rug
                   hover_data=dataset.columns)
fig.show()


# Conclusion:
# 
# 1. In comparison to bank balance, there are fewer outliers when it comes to annual salary. 
# 2. Default cases appear to be distributed across all annual salary ranges, suggesting that annual salary may not be a reliable predictor of loan defaults.

# Saving Rate distribution

# In[18]:


fig = px.histogram(dataset, x="Saving Rate",
                   color='Defaulted?', 
                   marginal="box", # or violin, rug
                   hover_data=dataset.columns)
fig.show()


# Conclusion:
# 
# The distribution of saving rate is similar to that of bank balance, but with a few extreme outliers. This suggests that people's saving habits can vary significantly. Some individuals may earn a high income but spend more than they save, while others with relatively low salaries may have a significant amount of savings.

# ## Modeling

# Train test split

# In[19]:


RAND_SEED = 123


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:,:-1], dataset.iloc[:,-1], test_size=0.3, stratify=dataset.iloc[:,-1], random_state=RAND_SEED)


# In[21]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Standardization

# In[22]:


scaler = StandardScaler().fit(X_train)


# In[23]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Upsampling by SMOTE

# During the Exploratory Data Analysis (EDA) phase, it was observed that defaulted cases constituted only 3% of the samples. 
# This highly imbalanced dataset could pose a challenge for classification models that aim to minimize the cost function. 
# To address this issue, the SMOTE upsampling method was introduced to rebalance the dataset.

# In[24]:


X_train.shape, y_train.shape


# In[25]:


y_train.value_counts()


# In[26]:


#pip install --upgrade imbalanced-learn

sm = SMOTE(random_state=RAND_SEED)
X_train, y_train = sm.fit_resample(X_train, y_train)


# In[27]:


X_train.shape, y_train.shape


# In[28]:


y_train.value_counts()


# ## Classification

# The models we will examine include </br>
# Logistic Regression, </br>
# Support Vector Machine, </br>
# Random Forest, LightGBM, and </br>
# XGboost. </br>
# Our primary metric for optimization is the Recall Rate for predicting defaulted cases. </br>
# This is because for a bank loan default problem, rejecting loans falsely only leads to potential interest loss, </br>
# while the default of a loan leads to a significant loss of all principal.</br>

# Logistic regression

# In[29]:


clf = LogisticRegression(solver='saga',random_state=RAND_SEED).fit(X_train, y_train)


# In[30]:


y_pred = clf.predict(X_test)


# Cross validation

# In[31]:


cross_val_score(clf, X_train, y_train, scoring='recall' ,cv=5, )


# First prediction result

# In[32]:


print(confusion_matrix(y_test,y_pred))


# In[33]:


print(classification_report(y_test,y_pred))


# Hyperparameter tuning

# In[34]:


from sklearn.model_selection import RandomizedSearchCV


# distributions = dict(C=np.linspace(2, 1000, 100),
#                      penalty=['l2', 'l1'])

# clf = RandomizedSearchCV(LogisticRegression(solver='saga',random_state=RAND_SEED), 
#                          distributions,
#                          scoring='recall', 
#                          n_iter=100,
#                          n_jobs = -1,
#                          random_state=RAND_SEED)
# clf_logistic = clf.fit(X_train, y_train)
# clf_logistic.best_params_

# {'penalty': 'l2', 'C': 254.02020202020202}

# In[35]:


distributions = dict(C=[254.02020202020202], penalty=['l2'])


# In[36]:


clf = RandomizedSearchCV(LogisticRegression(solver='saga',random_state=RAND_SEED), 
                         distributions,
                         scoring='recall', 
                         n_iter=100,
                         n_jobs = -1,
                         random_state=RAND_SEED)
clf_logistic = clf.fit(X_train, y_train)
clf_logistic.best_params_


# In[37]:


y_pred_logistic = clf_logistic.predict(X_test)


# Tuned prediction result

# In[38]:


print(confusion_matrix(y_test,y_pred_logistic))


# In[39]:


# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_logistic)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# In[40]:


print(classification_report(y_test,y_pred_logistic))


# Support vector machine

# In[41]:


clf = SVC(probability=True)
clf.fit(X_train, y_train)


# In[42]:


y_pred = clf.predict(X_test)


# Cross validation

# In[43]:


cross_val_score(clf, X_train, y_train, scoring='recall' ,cv=5, )


# First prediction result

# In[44]:


print(confusion_matrix(y_test,y_pred))


# In[45]:


print(classification_report(y_test,y_pred))


# Hyperparameter tuning

# distributions = dict(C=np.logspace(0, 4, 50),
#                      degree = np.linspace(1,10,1),
#                      class_weight = [None, 'balanced'],
#                     )

# In[46]:


distributions = dict(C=[494.1713361323833],
                     degree = [1.0],
                     class_weight = [None],
                    )


# In[47]:


# For training speed the iteration is set to 1.
# Given more time we can of course train more iters.
clf = RandomizedSearchCV(SVC(probability=True, cache_size = 1024*25), 
                         distributions,
                         scoring='recall', 
                         n_iter=1, 
                         n_jobs = 1,
                         random_state=RAND_SEED) 
clf_SVC = clf.fit(X_train, y_train)
clf_SVC.best_params_


# {'degree': 1.0, 'class_weight': None, 'C': 494.1713361323833}

# In[48]:


y_pred_SVC = clf_SVC.predict(X_test)


# Tuned prediction result

# In[49]:


print(confusion_matrix(y_test,y_pred_SVC))


# In[50]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_SVC)
# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# In[51]:


print(classification_report(y_test,y_pred_SVC))


# Random forest 

# In[52]:


clf = RandomForestClassifier()
clf.fit(X_train, y_train)


# In[53]:


y_pred = clf.predict(X_test)


# Cross validation

# In[54]:


cross_val_score(clf, X_train, y_train, scoring='recall' ,cv=5, )


# First prediction result

# In[55]:


print(confusion_matrix(y_test,y_pred))


# In[56]:


print(classification_report(y_test,y_pred))


# Hyperparameter tuning

# distributions = dict(n_estimators=np.arange(10, 500, 10),
#                      criterion=['gini', 'entropy'],
#                      max_depth = range(20),
#                      min_samples_split = range(2, 20),
#                      min_samples_leaf = range(3, 50),
#                      bootstrap = [True, False],
#                      class_weight = ['balanced', 'balanced_subsample']
#                     )

# 
# clf = RandomizedSearchCV(RandomForestClassifier(), 
#                          distributions,
#                          scoring='recall', 
#                          n_iter=20,
#                          n_jobs = 4,
#                          random_state=RAND_SEED)
# clf_random_forest = clf.fit(X_train, y_train)
# clf_random_forest.best_params_

# {'n_estimators': 490,
#  'min_samples_split': 14,
#  'min_samples_leaf': 5,
#  'max_depth': 8,
#  'criterion': 'gini',
#  'class_weight': 'balanced_subsample',
#  'bootstrap': False}

# In[58]:


distributions = dict(n_estimators=[490],
                     criterion=['gini'],
                     max_depth = [8],
                     min_samples_split = [14],
                     min_samples_leaf = [5],
                     bootstrap = [False],
                     class_weight = ['balanced_subsample']
                    )


# In[59]:


clf = RandomizedSearchCV(RandomForestClassifier(), 
                         distributions,
                         scoring='recall', 
                         n_iter=20,
                         n_jobs = 4,
                         random_state=RAND_SEED)
clf_random_forest = clf.fit(X_train, y_train)
clf_random_forest.best_params_


# In[60]:


y_pred_random_forest = clf_random_forest.predict(X_test)


# Tuned prediction result

# In[61]:


print(confusion_matrix(y_test,y_pred_random_forest))


# In[62]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_random_forest)
# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# In[63]:


print(classification_report(y_test,y_pred_random_forest))


# LightGBM

# In[64]:


clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)


# In[65]:


y_pred = clf.predict(X_test)


# Cross validation

# In[66]:


cross_val_score(clf, X_train, y_train, scoring='recall' ,cv=5, )


# First prediction result

# In[67]:


print(confusion_matrix(y_test,y_pred))


# In[68]:


print(classification_report(y_test,y_pred))


# Hyperparameter tuning

# distributions = {
#     'learning_rate': np.logspace(-5, 2, 50),
#     'num_leaves': np.arange(10, 100, 10),
#     'max_depth' : np.arange(3, 13, 1),
#     'colsample_bytree' : np.linspace(0.1, 1, 10),
#     'min_split_gain' : np.linspace(0.01, 0.1, 10),
# }

# clf = RandomizedSearchCV(lgb.LGBMClassifier(), 
#                          distributions,
#                          scoring='recall', 
#                          n_iter=100,
#                          n_jobs = 4,
#                          random_state=RAND_SEED)
# clf_lgb = clf.fit(X_train, y_train)
# clf_lgb.best_params_

# {'num_leaves': 60,
#  'min_split_gain': 0.030000000000000006,
#  'max_depth': 8,
#  'learning_rate': 0.07196856730011514,
#  'colsample_bytree': 0.7000000000000001}

# In[69]:


distributions = {
    'learning_rate': [0.07196856730011514],
    'num_leaves': [60],
    'max_depth' : [8],
    'colsample_bytree' : [0.7000000000000001],
    'min_split_gain' : [0.030000000000000006],
}


# In[70]:


clf = RandomizedSearchCV(lgb.LGBMClassifier(), 
                         distributions,
                         scoring='recall', 
                         n_iter=100,
                         n_jobs = 4,
                         random_state=RAND_SEED)
clf_lgb = clf.fit(X_train, y_train)
clf_lgb.best_params_


# In[71]:


y_pred_lgb = clf_lgb.predict(X_test)


# Tuned prediction result

# In[72]:


print(confusion_matrix(y_test,y_pred_lgb))


# In[73]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_lgb)
# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# In[74]:


print(classification_report(y_test,y_pred_lgb))


# XGBoost

# In[75]:


clf = XGBClassifier()
clf.fit(X_train, y_train)


# In[76]:


y_pred = clf.predict(X_test)


# Cross validation

# In[77]:


cross_val_score(clf, X_train, y_train, scoring='recall' ,cv=5, )


# First prediction result

# In[78]:


print(confusion_matrix(y_test,y_pred))


# In[79]:


print(classification_report(y_test,y_pred))


# Hyperparameter tuning

# distributions = { 
#     'n_estimators': np.arange(100, 1000, 100),
#     'max_depth':np.arange(2,10,1),
#     'learning_rate':np.logspace(-4, 1, 50), 
#     'subsample':np.linspace(0.1, 1, 10),
#     'colsample_bytree':np.linspace(0.1, 1, 10), 
# }

# clf = RandomizedSearchCV(XGBClassifier(), 
#                          distributions,
#                          scoring='recall', 
#                          n_iter=10,
#                          n_jobs = 4,
#                          random_state=RAND_SEED)
# clf_xgb = clf.fit(X_train, y_train)
# clf_xgb.best_params_

# {'subsample': 0.9,
#  'n_estimators': 600,
#  'max_depth': 8,
#  'learning_rate': 0.008685113737513529,
#  'colsample_bytree': 0.6}

# In[80]:


distributions = { 'n_estimators': [600],
                 'max_depth':[8], 
                 'learning_rate':[0.008685113737513529],
                 'subsample':[0.9],
                 'colsample_bytree':[0.6], }


# In[81]:


clf = RandomizedSearchCV(XGBClassifier(), 
                         distributions,
                         scoring='recall', 
                         n_iter=10,
                         n_jobs = 4,
                         random_state=RAND_SEED)
clf_xgb = clf.fit(X_train, y_train)
clf_xgb.best_params_


# In[82]:


y_pred_xgb = clf_xgb.predict(X_test)


# Tuned prediction result

# In[83]:


print(confusion_matrix(y_test,y_pred_xgb))


# In[84]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)
# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# In[85]:


print(classification_report(y_test,y_pred_xgb))


# Model assessment

# ROC curve

# In[86]:


sns.set()


# In[87]:


model_names = ['LogisticRegression','SVM', 'RandomForest','LightGBM','XGBoost']
models = [clf_logistic, clf_SVC, clf_random_forest, clf_lgb, clf_xgb]

plt.figure(figsize=(8, 6))

for name, model in zip(model_names, models):
    prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    model_auc = round(auc(fpr, tpr), 4)
    plt.plot(fpr,tpr,label="{}, AUC={}".format(name, model_auc))

random_classifier=np.linspace(0.0, 1.0, 100)
plt.plot(random_classifier, random_classifier, 'r--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Given the imbalanced nature of our dataset, our emphasis is on the precision-recall curve.
# Based on the test set outcome, it can be concluded that the Logistic regression model performed well.

# ## Conclusion

# The purpose of this notebook is to work with an imbalanced loan default dataset using multiple ML models. Our findings reveal that the Random Forest model achieved the highest Recall rate of 89% on the test set. However, the Logistic Regression model surpassed all other models with the top AUC score of 0.5238 in the precision-recall curve. With the addition of more features and feature engineering, there is potential to further enhance the results in the future.

# In[ ]:




