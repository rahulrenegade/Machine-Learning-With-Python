import pandas as pd
import matplotlib.pyplot as plt 
import math
import seaborn as sns
import numpy as np
suv_data=pd.read_csv("F:/Development/Machine Learning/suv-data/suv_data.csv")
print(suv_data.head(10))
print("the no of passengers in the list is"+str(len(suv_data.index)))
sns.countplot(x="Purchased",data=suv_data)
sns.countplot(x="Purchased",hue="Gender",data=suv_data)
suv_data['Age'].plot.hist()
suv_data.info()
suv_data['EstimatedSalary'].plot.hist(bins=50,figsize=(10,5))
print(suv_data.isnull())
print(suv_data.isnull().sum())
sns.heatmap(suv_data.isnull(),yticklabels=False,cmap="viridis")
plt.show()
sns.boxplot(x="Gender",y="Age",data=suv_data)
plt.show()
suv_data.drop("User ID",axis=1,inplace=True)
suv_data.columns
suv_data.head(10)
Gen=pd.get_dummies(suv_data['Gender'],drop_first=True)
print(Gen.head(5))
suv_data=pd.concat([suv_data,Gen],axis=1)
print(suv_data.head(5))
suv_data.drop("Gender",axis=1,inplace=True)
print(suv_data.head(10))
X=suv_data.iloc[:,[0,1,3]].values
y=suv_data.iloc[:,2].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train, y_train)
predictions=logmodel.predict(X_test)
print(predictions)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions)*100)
