
# coding: utf-8

# In[666]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[667]:


dataset = pd.read_csv("/Users/karansingh/Documents/CS505 Random Forests/indian_liver_patient.csv")


# In[668]:


mod_dataset = dataset

mod_dataset['Gender'] = mod_dataset['Gender'].map({"Male": int(1), "Female" : int(0)})

mod_dataset['Dataset'] = mod_dataset['Dataset'].map({int(1): int(1), int(2): int(0)})

mod_dataset.isnull().any()


# In[669]:


mod_dataset.head()

mod_dataset.replace(r'\s+', np.nan, regex=True)
#mod_dataset = mod_dataset.dropna()


# In[670]:


# Prepare Data For Training
X = mod_dataset.iloc[:, 0:10].values

imp = Imputer(missing_values=np.nan, strategy='mean')

imp.fit(X)

X = imp.transform(X)


# In[671]:


print(X[209,:])
X


# In[672]:


Y = mod_dataset.iloc[:,10].values


# In[673]:


# The following code divides data into training and testing sets


# In[674]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state =40)


# In[675]:


#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[676]:


# Training the algorithim
classifier = RandomForestClassifier(n_estimators=700, random_state=0, max_features = 'auto')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[677]:


print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test, y_pred))

