# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 23:16:16 2019

@author: nikil reddy"""

#RELATIVE GRADE CLUSTERING 

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing datasets and manipulating
dsp=pd.read_csv('pass.csv')
dsf=pd.read_csv('FAIL.csv')
x=dsp.iloc[:,2:].values
xf=dsf.iloc[:,2:].values
x=np.append(arr=np.ones((150,1)).astype(int),values=x , axis=1 )
xf=np.append(arr=np.ones((50,1)).astype(int),values=xf , axis=1 )

#fitting model with data
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=6,init='k-means++',random_state=0)

#predicting clusters
y_kmeans=kmeans.fit_predict(x)

#Visualising the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],color='orange',label='A')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],color='blue',label='D')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],color='green',label='P')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],color='yellow',label='B')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],color='cyan',label='C')
plt.scatter(x[y_kmeans==5,0],x[y_kmeans==5,1],color='magenta',label='ex')
plt.scatter(xf[:,0],dsf.iloc[:,2],color='red',label='FAIL')
plt.title('relative_grade clustering')
plt.ylabel('marks')
plt.legend()
plt.show()

