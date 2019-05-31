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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=400,init='uniform',activation='relu',input_dim=784))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(output_dim=200,init='uniform',activation='relu'))
    classifier.add(Dropout(p=0.1))  
    classifier.add(Dense(output_dim=100,init='uniform',activation='relu'))
    classifier.add(Dropout(p=0.1)) 
    classifier.add(Dense(output_dim=50,init='uniform',activation='relu'))
    classifier.add(Dropout(p=0.1)) 
    classifier.add(Dense(output_dim=10,init='uniform',activation='softmax'))
    classifier.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[200,300],
              'epochs':[2,4],
              'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters  """  ,scoring='accuracy'  """  ,cv=10)
grid_search = grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

testing = pd.read_csv('test.csv')
tt = testing.iloc[:,:].values
pred = grid_search.predict(tt)

pred1 = pd.DataFrame(pred)
pred1 = pd.DataFrame(pred1.idxmax(axis = 1))
pred1.index.name = 'ImageId'
pred1 = pred1.rename(columns = {0: 'Label'}).reset_index()
pred1['ImageId'] = pred1['ImageId'] + 1
pred1.to_csv('predictedgs.csv', index = False)