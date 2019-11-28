#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv(r"C:\Users\BISWA\Desktop\ML Project\Kaggle Comp\train.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())


# In[9]:



df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[10]:


df.drop(['Alley'],axis=1,inplace=True)
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
df.drop(['Id'],axis=1,inplace=True)
df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[11]:


df.shape


# In[12]:


df.isnull().sum()


# In[13]:


df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])


# In[14]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[15]:


df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[16]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')


# In[17]:


df.dropna(inplace=True)


# In[18]:


df.shape


# In[19]:


df.head()


# In[20]:


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


# In[21]:



len(columns)


# In[22]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[23]:


main_df=df.copy()


# In[24]:


df.head()


# In[25]:


df['SalePrice']


# In[26]:


test_df=pd.read_csv('formulatedtest.csv')


# In[27]:


test_df.shape


# In[28]:



test_df.head()


# In[29]:


final_df=pd.concat([df,test_df],axis=0)


# In[30]:


final_df.shape


# In[31]:


final_df=category_onehot_multcols(columns)


# In[32]:


final_df.shape


# In[33]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[34]:


final_df.shape


# In[35]:



df_Train=final_df.iloc[:1436,:]
df_Test=final_df.iloc[1436:,:]


# In[36]:


df_Train.head()


# In[37]:


df_Test.head()


# In[38]:


df_Train.shape


# In[39]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[40]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# In[42]:


import xgboost
classifier = xgboost.XGBRegressor()
classifier.fit(X_train,y_train)


# In[43]:


y_pre = classifier.predict(df_Test)


# In[44]:


y_pre


# In[46]:


pred = pd.DataFrame(y_pre)


# In[49]:


sub_df = pd.read_csv(r'C:\Users\BISWA\Desktop\ML Project\Kaggle Comp\sample_submission.csv')


# In[52]:


datasets = pd.concat([sub_df['Id'],pred], axis = 1)


# In[53]:


datasets.columns = ['Id','Salesprice']


# In[54]:


datasets.to_csv('sample Submission.csv', index=False)


# In[ ]:




