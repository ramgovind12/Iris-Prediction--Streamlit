import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv("Iris.csv")
df.head()

X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df[['Species']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 1)

model = SVC(gamma = 'auto')
sv = model.fit(X_train,y_train)

#predictions = model.predict(X_test)

import joblib
joblib.dump(sv,'svm_file.pkl')
