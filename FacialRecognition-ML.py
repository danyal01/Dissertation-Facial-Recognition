#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# # Dataset

# In[2]:


# Converting all the known Images to grey images
path = 'KnownImages'
myList = os.listdir(path)

for i in myList:
    openedImg = Image.open(f'{path}/{i}')
    studentNames = os.path.splitext(i)[0]
    newName = studentNames + 'grey'
    openedImg = openedImg.resize((64, 64)).convert('L')
    openedImg.save('GreyedImages/' + newName + '.png')


# In[3]:


# Encoding thes images array and adding them to the dataset
path = 'GreyedImages'
myList = os.listdir(path)    
images_greyed = []
count = 0 

with open(r'datasets\olivetti_X.csv', 'a', newline='') as f:
    for i in myList:
        count = count + 1
        im = img.imread(f'{path}/{i}')
        image_reshape = (im.flatten().reshape(-1, 1).T)
        writer = csv.writer(f)        
        writer.writerows(image_reshape)
        
        lastRow = pd.read_csv("datasets\olivetti_y.csv").iloc[-1][0]
        data = [str(lastRow+1)]
        with open(r'datasets\olivetti_y.csv', 'a', newline='') as file:
            writer = csv.writer(file)  
            writer.writerow(data)
file.close()
f.close()


# In[4]:


# Reading the updated datasets as array
data = np.loadtxt(open("datasets\olivetti_X.csv", "rb"), delimiter=",", skiprows=1)
target = np.loadtxt(open("datasets\olivetti_Y.csv", "rb"), delimiter=",", skiprows=1)
data


# # Data Anyalysis

# In[5]:


print("There are {} images in the dataset".format(len(data)))
print("There are {} unique targets in the dataset".format(len(np.unique(target))))
print("There are {} input features".format(data.shape[1]))
print("Size of each image is 64 x 64")


# # Data Pre-processing 

# # Spliting Dataset

# In[6]:


X = data
y = target
Height = 64
Width = 64
# Split into a training set (75%) and a test set (25%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# # PCA

# In[7]:


pca = PCA(n_components=150, whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Current shape of input data matrix: ", X_train_pca.shape)


# # KNeighborsClassifier

# In[8]:


model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train_pca, y_train)


# In[9]:


accuracy = model.score(X_test_pca,y_test)
print("Testing Score = {:.3f}".format(accuracy))


# In[10]:


accuracy = model.score(X_train_pca,y_train)
print("Training Score = {:.3f}".format(accuracy))


# # Confusion Matrix

# In[11]:


y_test_pred = model.predict(X_test_pca)
cfm = confusion_matrix(y_test,y_test_pred)
cfm


# # Evaluation Metrics

# In[12]:


cReport = classification_report(y_test, y_test_pred)
print(cReport)


# # Resetting Datasets

# In[13]:


df = pd.read_csv('datasets/olivetti_X.csv')
df.drop(df.tail(count).index,inplace=True)
df.to_csv('datasets\olivetti_X.csv', index=False)


# In[14]:


df = pd.read_csv('datasets/olivetti_y.csv')
df.drop(df.tail(count).index,inplace=True)
df.to_csv('datasets\olivetti_y.csv', index=False)


# In[ ]:




