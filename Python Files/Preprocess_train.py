import pickle
import numpy as np #for numerical data
import pandas as pd #for Performing operations on data
from sklearn.metrics import accuracy_score #for generating accuracy on test data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:\\Users\\Vaibhav\\Desktop\\Machine Learning Dataset Implementation\\machine.csv")
y=df.vendor_name.astype("category").cat.codes  #converting categorical label into numerical 
col = list(df.columns)[:2] 
X=df.drop(col,axis=1) #data set for training

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20)  #separing train test to 80% and 20%
clf=RandomForestClassifier() #building random forest classifier
clf.fit(X_train,y_train)  #model using training set
clf.score(X_train,y_train)
predicted=clf.predict(X_test) #prediction on test set
accuracy_score(y_test, predicted) #Accuracy of prediction

# save the model to disk
filename = 'finalized_Prediction_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
result
