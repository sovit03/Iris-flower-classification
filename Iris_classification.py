#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score 


# In[3]:


url= "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_df = pd.read_csv(url, header = None)


# In[4]:


iris_df


# In[9]:


column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width","class"]


# In[ ]:





# In[10]:


iris_df.columns = column_names


# In[11]:


iris_df


# In[13]:


sns.pairplot(iris_df,hue="class")
plt.show()


# In[14]:


iris_df.describe()


# In[15]:


'''iris_df.groupby("class")["Iris-setosa"].describe()


# In[16]:


iris_df[iris_df["class"] == "Iris-setosa"].describe()


# In[19]:


iris_df[iris_df["class"] == "Iris-virginica"].describe()


# iris_df1 = [iris_df["class"] == "Iris-virginica"]

# In[22]:


iris_df1 = iris_df[iris_df["class"] == "Iris-virginica"]


# In[23]:


iris_df1


# In[25]:


iris_df[95:150]


# In[26]:


iris_df1.count()'''


# In[27]:





# In[28]:


print(sklearn.__version__)


# In[31]:





# In[33]:


a = iris_df.drop("class", axis=1)


# In[34]:


a


# In[35]:


b = iris_df["class"]


# In[36]:


b


# In[37]:


a_train, a_test, b_train, b_test = train_test_split(a,b,test_size = 0.3, random_state =42 )


# In[39]:


knn = KNeighborsClassifier( n_neighbors = 3)
knn.fit(a_train, b_train)


# In[41]:


b_pred = knn.predict(a_test)
print("Accuracy:",accuracy_score(b_test,b_pred))


# In[42]:


print(classification_report(b_test,b_pred))


# In[ ]:




