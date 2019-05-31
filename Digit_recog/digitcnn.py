import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:785].values
y = dataset.iloc[:, 0:1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
y[:, 0] = labelencoder_X_1.fit_transform(y[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = X_train.reshape(X_train.shape[0],28, 28,1).astype('float64')
X_test = X_test.reshape( X_test.shape[0],28, 28,1).astype('float64')

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

classifire = Sequential()
classifire.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))
classifire.add(MaxPooling2D(pool_size=(2,2)))
classifire.add(Conv2D(15,(3,3),input_shape=(28,28,1),activation='relu'))
classifire.add(MaxPooling2D(pool_size=(2,2)))
classifire.add(Flatten())
classifire.add(Dense(output_dim=128,activation='relu'))
classifire.add(Dense(output_dim=50,activation='relu'))
classifire.add(Dense(output_dim=10,activation='softmax'))
classifire.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifire.fit(X_train,y_train,batch_size=200,epochs=10)

testing = pd.read_csv('test.csv')
tt = testing.iloc[:,:].values
tt = tt.reshape( tt.shape[0],28, 28,1).astype('float64')
pred = classifire.predict(tt)

pred1 = pd.DataFrame(pred)
pred1 = pd.DataFrame(pred1.idxmax(axis = 1))
pred1.index.name = 'ImageId'
pred1 = pred1.rename(columns = {0: 'Label'}).reset_index()
pred1['ImageId'] = pred1['ImageId'] + 1
pred1.to_csv('predictedcnn.csv', index = False)