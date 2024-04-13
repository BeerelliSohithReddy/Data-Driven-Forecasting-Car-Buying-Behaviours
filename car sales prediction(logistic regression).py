import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
dataset=pd.read_csv(r'D:\data science class\Projects\Future Predicition of Buying a Car\car prediction.csv')

y=dataset['Purchased']

x=dataset[['Age','EstimatedSalary']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.20,random_state=4)


from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()
x_train=mn.fit_transform(x_train )
x_test=mn.transform(x_test)

from sklearn.linear_model import LogisticRegression
log=LogisticRegression(penalty="elasticnet",solver='saga',l1_ratio=1)
log.fit(x_train, y_train)
y_pred=log.predict(x_test)
y_pred

from sklearn.model_selection import cross_val_score
score=cross_val_score(log, x_train,y_train,cv=5)
print(score)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred,)
print(cr)

from sklearn.metrics import accuracy_score
ac=accuracy_score( y_test, y_pred,)
print(ac)


bias=log.score(x_train, y_train)
bias
variance=log.score(x_test,y_test)
variance