#!/usr/bin/env python
# coding: utf-8

# # Unit 2: Supervised Learning Project

# In[4]:


import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn


# ## 1. Import libraries
# 
# Import all of the modules, functions, and objects we will use in this tutorial.

# In[5]:


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ## 2. Load the Dataset
# 
# We will be using the iris flowers dataset, which contains 150 observations of iris flowers. There are four columns of measurements and the species of flower observed.  Only three species are present in this dataset.
# 
# The data can be loaded directly from the UCI Machine Learning Repository

# In[6]:


# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# ## 2.1 Dataset Properties
# 
# Lets take a look at the dataset by observing its dimensions, the first few rows of data, a statistical summary of the attributes, and a breakdown of the data by the class variable.

# In[7]:


# Shape
print(dataset.shape)


# In[8]:


# Head
print(dataset.head(20))


# In[9]:


# descriptions
print(dataset.describe())


# In[10]:


# class distribution
print(dataset.groupby('class').size())


# ## 2.2 Data Visualizations
# 
# Lets visualize the data so we can understand the distribution of the input attributes. We will use histograms of each attribute, as well as some multivariate plots so that we can view the interactions between variables.

# In[11]:


# histograms
dataset.hist()
plt.show()


# In[12]:


# scatter plot matrix
scatter_matrix(dataset)
plt.show()


# ## 3. Evaluate Algorithms
# 
# Lets create some models of the data and estimate their accuracy on unseen data.
# 
# We are going to,
# 
# * Create a validation dataset
# * Set-up cross validation
# * Build three different models to predict species from flower measurement
# * Select the best model
# 
# ## 3.1 Create Validation Dataset
# 
# Lets split the loaded dataset into two.  80% of the data will be used for training, while 20% will be used for validation.

# In[13]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


# ## 3.2 10-fold Cross Validation
# 
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeate for all combinations of train-test splits

# In[14]:


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# ## 3.3 Build Models
# 
# Lets evaluate three models:
# 
# * Logistic Regression (LR)
# * K-Nearest Neighbors (KNN)
# * Support Vector Machine (SVM)

# In[15]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# ## 4. Make Predictions
# 
# Lets test the model on the validation set to make sure that our algorithms can generalize to new data.  Otherwise, we may be overfitting the training data.  

# In[16]:


# Make predictions on validation dataset

for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(name)
    print(accuracy_score(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


# In[ ]:





# In[ ]:




