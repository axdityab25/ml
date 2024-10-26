import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


iris=load_iris()


X =iris.data
Y=iris.target


X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


Classifier=RandomForestClassifier(n_estimators=100,random_state=0)
Classifier.fit(X_train,Y_train)


y_pred=Classifier.predict(X_test)


accuracy=accuracy_score(Y_test,y_pred)


print(accuracy)


pca=PCA(n_components=2)
x_train=pca.fit_transform(X_train)
x_test=pca.transform(X_test)


Classifier1=RandomForestClassifier(n_estimators=100,random_state=0)
Classifier1.fit(x_train,Y_train)


y_pred1=Classifier1.predict(x_test)


accuracy1=accuracy_score(Y_test,y_pred1)
print(accuracy1)
