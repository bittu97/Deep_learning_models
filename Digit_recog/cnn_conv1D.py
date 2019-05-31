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

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
X_test = np.reshape(X_test, ( X_test.shape[0], X_test.shape[1],1))

"""X_train = X_train.reshape(28, 28,1).astype('float64')
X_test = X_test.reshape(28, 28,1).astype('float64')"""
 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten

classifire = Sequential()
classifire.add(Conv1D(32,4,input_shape=(X_train.shape[1],X_train.shape[2] ),activation='relu'))
classifire.add(MaxPooling1D(pool_size=(3)))
classifire.add(Flatten())
classifire.add(Dense(output_dim=128,activation='relu'))
classifire.add(Dense(output_dim=10,activation='softmax'))
classifire.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifire.fit(X_train,y_train,batch_size=200,epochs=10)

testing = pd.read_csv('test.csv')
tt = testing.iloc[:,:].values
tt = tt.reshape(tt.shape[0], tt.shape[1],1).astype('float64')
pred = classifire.predict(tt)

pred1 = pd.DataFrame(pred)
pred1 = pd.DataFrame(pred1.idxmax(axis = 1))
pred1.index.name = 'ImageId'
pred1 = pred1.rename(columns = {0: 'Label'}).reset_index()
pred1['ImageId'] = pred1['ImageId'] + 1
pred1.to_csv('predictedcnn1D.csv', index = False)