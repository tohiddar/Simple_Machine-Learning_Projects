#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# #### through
# 
# * Local Outlier Factor (LOF)
# * Isolation Forest Algorithm

# In[1]:


# import the necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# # Sections
# 
# #### 1. Dataset Loading and Analysis
# #### 2. Data Visualization
# #### 3. Feature Selection
# #### 4. Model Training & Testing
# #### 5. Performance comparison between LOF and IF using evaluation metrics

# ### 1. Dataset Loading and Analysis

# In[2]:


# Load Dataset into a dataframeframe using pandas
dataframe = pd.read_csv('creditcard.csv')


# In[3]:


print("Shape of the Dataset: ", dataframe.shape) # number of rows and columns in our dataset
print("\n\n", dataframe.columns) # columns/features in our Dataset


# In[4]:


dataframe.head() # first five records


# In[5]:


dataframe.tail() # last five records


# In[6]:


# Print the shape of the dataframe

dataframe = dataframe.sample(frac = 0.3, random_state = 42) # using 30% of our dataset for next steps
print("Shape of the Dataset: ", dataframe.shape)


# In[7]:


# Determine number of fraud cases in Dataset

Fraud = dataframe[dataframe['Class'] == 1]
Valid = dataframe[dataframe['Class'] == 0]

outlier_fraction = (len(Fraud)/float(len(Valid)))
print("Outlier_fraction: {0} %".format(outlier_fraction*100))

print('Fraud Cases: {}'.format(len(dataframe[dataframe['Class'] == 1])))
print('Valid Transactions: {}'.format(len(dataframe[dataframe['Class'] == 0])))


# In[8]:


print("Description of the Dataset: ", dataframe.describe())

# The columns have been encrypted using PCA Dimensionality reduction to protect user identities and sensitive features


# ### 2. dataframe Visualization

# In[9]:


# Plot histograms for each parameter 

dataframe.hist(figsize = (15, 15))
plt.show()


# In[10]:


# Correlation matrix

corrmat = dataframe.corr()
fig = plt.figure(figsize = (15, 15))

#Plotting a heatmap to visualize the correlation matrix and see features 
# with strong correlation to the target class
sns.heatmap(corrmat, vmax = .6, square = True) # vmax is the max and min value you want to have for the scale
plt.show()


# In[11]:


corrmat['Class']


# In[12]:


len(corrmat['Class'])


# ### 3. Feature Selection

# In[13]:


# getting columns which have correlation score > 0.01 or < -0.01, you can chose a different constant and experiment
cols = corrmat.keys()
cols_to_keep = []

for i in range(len(corrmat)):
    
    if abs(corrmat['Class'][i]) > 0.01:
        
        cols_to_keep.append(cols[i])


# In[14]:


len(cols_to_keep) # the final features list we wish to keep


# In[15]:


cols_to_keep


# In[16]:


# removing the 'Class' columnn from the features list, as it is the variable we wish to predict
cols = cols_to_keep[:-1]


# In[17]:


features = dataframe[cols] # records of all transactions, excluding the target class
target = dataframe["Class"] # records of the corresponding label for each record

print(features.shape)
print(target.shape)


# ### 4. Model Training and Testing
# 
# The machine learning algorithms that we are going to use are:
# 
# 1. Local Outlier Factor
# 2. Isolation Forest
# 
# We are using them by importing them directly from scikit-learn; however, if you're interested in learning about their theory, you can refer to the two resources mentioned below:
# 
# ##### LOF: https://towardsdataframescience.com/local-outlier-factor-for-anomaly-detection-cc0c770d2ebe
# 
# ##### IF: https://medium.com/@often_weird/isolation-forest-algorithm-for-anomaly-detection-f88af2d5518d

# In[19]:


# define random states
state = 1

# define outlier detection tools to be compared
classifiers = {
    "IF": IsolationForest(max_samples = len(features),
                                        contamination = outlier_fraction,
                                        random_state = state),
    "LOF": LocalOutlierFactor(
        n_neighbors = 20,
        contamination = outlier_fraction)}


# In[20]:


# skipping the train, test split step because we wish the model to overfit on these features and learn 
# a mathematical function to map the features

n_outliers = len(Fraud)

# Fit the model
for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the dataframe and tag outliers
    if clf_name == "LOF":
        
        y_pred = clf.fit_predict(features)
        scores_pred = clf.negative_outlier_factor_
        
    else:
        
        # train/fit classifier on our features
        clf.fit(features)
        # generate predictions 
        scores_pred = clf.decision_function(features)
        y_pred = clf.predict(features)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud.
    
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != target).sum()
    
    # Run classification metrics
    print('Classifier {0}: \nNumber of Errors: {1}'.format(clf_name, n_errors))
    print('Accuracy: {0}%\n'.format(accuracy_score(target, y_pred)*100))
    print(classification_report(target, y_pred))

