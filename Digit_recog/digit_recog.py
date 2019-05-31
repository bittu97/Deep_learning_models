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

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

input_d = X_train.shape[1]

classifier = Sequential()
classifier.add(Dense(output_dim=128,init='uniform',activation='relu',input_dim=input_d))
classifier.add(Dense(output_dim=64,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax'))
#classifier.add(Dense(output_dim=1,init='uniform',activation='softmax'))

classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=64, epochs=25)