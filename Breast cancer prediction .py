#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Data load

# In[3]:


from sklearn. datasets import load_breast_cancer 
cancer_dataset = load_breast_cancer()


# # Data manipulations

# In[4]:


cancer_dataset


# In[5]:


type(cancer_dataset)


# In[6]:


#key in data set 
cancer_dataset.keys()


# In[7]:


cancer_dataset["data"]


# In[8]:


cancer_dataset["target"]


# In[9]:


cancer_dataset["target_names"]


# In[10]:


#print description data
print(cancer_dataset["DESCR"])


# In[11]:


#NAME OF FEATURE
print(cancer_dataset["feature_names"])


# In[12]:


#locate/path of the data file 
print(cancer_dataset["filename"])


# # create DataFrame

# In[13]:


#create data frame
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
                       columns= np.append(cancer_dataset['feature_names'],['target']))


# In[14]:


#data frame to csv file
cancer_df.to_csv("breast_cancer_dataframe.csv")


# In[15]:


#head of cancer datfarem
cancer_df.head(7)


# In[16]:


cancer_df.tail(7)


# In[17]:


cancer_df.info()


# In[18]:


#b=numarical distrubtion of data 
cancer_df.describe()


# In[19]:


cancer_df.isnull().sum()


# # Data visualization

# In[20]:


#count the target  class


# In[21]:


sns.countplot(cancer_df["target"])


# In[22]:


#counter plot of feature mean radus
plt.figure(figsize=(20,8))
sns.countplot(cancer_df["mean radius"])


# # Heatmap

# In[23]:


#heat map Dataframe
plt.figure(figsize = (16,8))
sns.heatmap(cancer_df)
    


# # Heatmap of corelation matrix

# In[24]:


cancer_df.corr()


# In[25]:


#Heatmap of correlation matrix of Brest cancer DataFrame
plt.figure(figsize=(20,10))
sns.heatmap(cancer_df.corr(),annot = True, cmap="coolwarm",linewidth = 2)


# # Corrleation of Barplot

# In[26]:


# create seconf Dataframe by droping target 
cancer_df2 = cancer_df.drop(["target"],axis= 1)


# In[27]:


print("The shape of cancer_df2 is :", cancer_df.shape)


# In[28]:


#cancer_df2.coeewith(cancer_df.target )


# In[29]:


#visualtization of correlation of barplot 
#plt.figure(figsize=(16,5))
#ax = sns.barplot(cancer_df2.corrwitdth(cancer_df.target).indez,cancer_df2.corrwidth(cancer_df.target 
#ax.tick_params(label(ratation = 90)


# In[30]:


cancer_df2.corrwith(cancer_df.target).index


# In[31]:


plt.figure(figsize = (20,20))
sns.heatmap(cancer_df.corr(),annot = False ,cmap='coolwarm',linewidth=4)


# # Correlation Barplot

# In[32]:


#create second DataFrame bhy drawing by drop target 
cancer_df2 = cancer_df.drop(["target"], axis =   1)


# In[33]:


print("The shape of the 'cancer_df2' is : ",cancer_df2.shape)


# In[34]:


#cancer df2.corrrwith(cancer df.target)


# In[35]:


cancer_df2.corrwith(cancer_df.target).index


# # split DataFrame in Train and Test 

# In[36]:


#input variable 
x = cancer_df.drop(["target"], axis = 1)
x.head(7)


# In[37]:


# output varibale 
y = cancer_df["target"]
y.head(7)


# In[38]:


#split dataset into train and test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 5)


# In[39]:


x_train


# In[40]:


x_test


# In[41]:


y_train


# In[42]:


y_test


# # Feature scaling 

# In[80]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


# # Maching learning Bulding model 

# # suppor vector classifer accuray. 98%

# In[81]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[82]:


# support vector


# In[83]:


svc_classifier2 = SVC()
svc_classifier2.fit(x_train, y_train)
y_pred_scv = svc_classifier2.predict(x_test)
accuracy_score(y_test,y_pred_scv)


# In[78]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


# In[52]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[53]:


print('Shape for training data', x_train.shape, y_train.shape)
print("_________________________________")
print('Shape for testing data', x_test.shape, y_test.shape)


# In[54]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# # Random Forest Regressor Model .. acc. 85.78%

# In[55]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=1000)


# In[56]:


model.fit(x_train,y_train)


# In[57]:


print("Accuracy --> ", model.score(x_test, y_test)*100)


# # Gradient Boosting Regressor Model .. acc.85.82%

# In[58]:


from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)


# In[59]:


model.fit(x_train,y_train)


# In[60]:


print("Accuracy --> ", model.score(x_test, y_test)*100)

