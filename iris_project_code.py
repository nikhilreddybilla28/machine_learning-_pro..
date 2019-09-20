import pandas as pd
import matplotlib.pyplot as plt

#importing data
ds=pd.read_csv('IRIS.csv')
x=ds.iloc[:,0:4].values
y=ds.iloc[:,4]

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
y=le.fit_transform(y)

#splitting data into training_set and test_set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_tet=train_test_split(x,y,test_size=0.25,random_state=28)
'''random_state can be any value it is uesd to get same results every time, because,
splitting of datapoints will be same'''

#fitting classifier with training data 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

#predicting test results
y_pred=knn.predict(x_test)

#accuracy_score
from sklearn.metrics import accuracy_score
accuracy_score(y_tet,y_pred)

#visualising the test results
y=ds.iloc[:,4]
plt.scatter(ds.iloc[:,0],y,color='red',label='Sepal_Length')
plt.scatter(ds.iloc[:,1],y,color='blue',label='Sepal_Width')
plt.scatter(ds.iloc[:,2],y,color='green',label='Petal_Length')
plt.scatter(ds.iloc[:,3],y,color='cyan',label='Petal_Width')
plt.xlabel('features')
plt.ylabel('class')
plt.legend()
plt.show()

'''here we can get acuracy=1 , which i got by chaning random state and n_neighbours,but i got precise and
accurate for n_neighbors=3'''  