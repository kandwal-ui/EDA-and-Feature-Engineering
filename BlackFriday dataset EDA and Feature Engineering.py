#!/usr/bin/env python
# coding: utf-8

# # Importing the required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Problem statement
#   
#    A retail company 'ABC private limited' wants to   understand the customer purchase behaviour(specifically purchase amount) aginst various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month . The dataset also contains customer demographics (age, gender, marital status,city_type,stay_in_current_city), product details(product_id and product_category) and total purchase_amount from last month.
#    
#    Now they want to build a model to predict the purchase amount of customer against products which will help them to create personalized offer for customers against different products.

# # Import the Dataset

# In[2]:


train_df=pd.read_csv('train.csv')
train_df.head()


# In[3]:


# Importing the test dataset
test_df=pd.read_csv('test.csv')
test_df.head()


# In[4]:


# merge both train and test dataset
df=train_df.append(test_df)
df.head()


# In[5]:


df.info()


# In[6]:


# find statistical info
df.describe()


# In[7]:


# drop unwanted columns
df.drop(['User_ID'],axis=1,inplace=True)


# In[8]:


df.head()


# In[9]:


# handling the categorical column gender
df['Gender']=df['Gender'].map({'F':0,'M':1})
df.head()


# In[10]:


# handle categorical feature age
df['Age'].unique()


# In[11]:


df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7}) 


# In[12]:


df.head()


# In[13]:


#  second technique
# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows 
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['Age']= label_encoder.fit_transform(df['Age'])
  
df['Age'].unique()


# In[14]:


# transform categorical column City_category
df_city=pd.get_dummies(df['City_Category'],drop_first=True)


# In[15]:


df_city.head()


# In[16]:


df=pd.concat([df,df_city],axis=1)
df.head()


# In[17]:


# drop city category column
df.drop('City_Category',axis=1,inplace=True)


# In[18]:


df.head()


# In[19]:


# handling missing values
df.isnull().sum()


# In[20]:


# replace the missing values with mode in Product_Category_2
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[21]:


df['Product_Category_2'].isnull().sum()


# In[22]:


# replace the missing values with mode in Product_Category_3
df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[23]:


df['Product_Category_3'].isnull().sum()


# In[24]:


df.head()


# In[25]:


df['Stay_In_Current_City_Years'].unique()


# In[26]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')


# In[27]:


df.head()


# In[28]:


# convert object into integers
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)


# In[29]:


df.info()


# In[30]:


df['B']=df['B'].astype(int)
df['C']=df['C'].astype(int)


# In[31]:


df.info()


# In[32]:


# Visualisation
sns.barplot('Age','Purchase',hue='Gender',data=df)


# # Purchasing of men is higher than women

# In[33]:


# visualisation of purchase with occupation
sns.barplot('Occupation','Purchase',hue='Gender',data=df)


# In[34]:


sns.barplot('Product_Category_1','Purchase',hue='Gender',data=df)


# In[35]:


sns.barplot('Product_Category_2','Purchase',hue='Gender',data=df)


# In[36]:


sns.barplot('Product_Category_3','Purchase',hue='Gender',data=df)


# In[37]:


df.head()


# In[38]:


# Feature scaling
test_df=df[df['Purchase'].isnull()]


# In[39]:


train_df=df[~df['Purchase'].isnull()]


# In[48]:


X=train_df.drop('Purchase',axis=1)


# In[49]:


X.head()


# In[51]:


X.shape


# In[46]:


y=train_df['Purchase']


# In[47]:


y.head()


# In[55]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)


# In[56]:


X_train.drop('Product_ID',axis=1,inplace=True)
X_test.drop('Product_ID',axis=1,inplace=True)


# In[57]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:


# training the model

