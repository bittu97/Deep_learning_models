import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,[3,4,5,6,7,8,10,11,12,13,15,16,17,18,19,20,21,22,23]].values
y = dataset.iloc[:, 24:25].values

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
y[:, 0] = labelencoder_X_1.fit_transform(y[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
classifier = Sequential()
classifier.add(Dense(output_dim=15,init='uniform',activation='relu',input_dim=19))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=10,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1)) 
classifier.add(Dense(output_dim=5,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))  
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
"""classifier = Sequential()
classifier.add(Dense(output_dim=10,init='uniform',activation='relu',input_dim=19))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=200,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))  
classifier.add(Dense(output_dim=100,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1)) 
classifier.add(Dense(output_dim=50,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1)) 
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax'))"""

classifier.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=500, epochs=10)
pred1 =  classifier.predict(X_test)
r2=r2_score(y_test, pred1)
print("r2 score value " , r2)

testing = pd.read_csv('test.csv')
tt = testing.iloc[:,[5,6,7,8,9,10,12,13,14,15,17,18,19,20,21,22,23,24,25]].values
sc1 = StandardScaler()
tt = sc.fit_transform(tt)
pred = classifier.predict(tt)

g = testing.iloc[:,:].values

pred1 = pd.DataFrame(pred)
#pred1 = pd.DataFrame(pred1.idxmax(axis = 1))
pred1.index.name = 'soldierId'
pred1 = pred1.rename(columns = {0: 'bestSoldierPerc'}).reset_index()
pred1['soldierId'] = g[:,2]
pred1.to_csv('predictedf.csv', index = False)