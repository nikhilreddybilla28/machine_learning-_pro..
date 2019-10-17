#RELATIVE GRADE CLUSTERING 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds=pd.read_csv('Mall_Customers.csv')
x=ds.iloc[:,4:5].values
x=np.append(arr=np.ones((200,1)).astype(int),values=x , axis=1 )

from sklearn.cluster import KMeans
#Applying k-means to dataset
kmeans=KMeans(n_clusters=6,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x)

#Visualising the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],color='red',label='A')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],color='blue',label='D')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],color='green',label='P')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],color='yellow',label='B')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],color='cyan',label='C')
plt.scatter(x[y_kmeans==5,0],x[y_kmeans==5,1],color='magenta',label='ex')
plt.title('relative_grade clustering')
plt.ylabel('marks')
plt.legend()
plt.show()
