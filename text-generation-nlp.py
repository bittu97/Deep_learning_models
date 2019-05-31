import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, CuDNNLSTM, Dropout
from keras.utils import to_categorical
from random import randint

print(os.listdir("../input"))

file = open('../input/Ancient_Modern_Physics.txt','r')
text = file.read()
file.close()
text[:1000]

tokens = text.lower()
print(tokens[:500])

n_chars = len(tokens)
unique_vocab = len(set(tokens))
print('Total Tokens: %d' % n_chars)
print('Unique Tokens: %d' % unique_vocab)

characters = sorted(list(set(tokens)))
n_vocab = len(characters)
n_vocab

int_to_char = {n:char for n, char in enumerate(characters)}
char_to_int = {char:n for n, char in enumerate(characters)}

X = []
y = []
seq_length = 100

for i in range(0, n_chars - seq_length, 1):
    seq_in = tokens[i:i + seq_length]
    seq_out = tokens[i + seq_length]
    X.append([char_to_int[char] for char in seq_in])
    y.append(char_to_int[seq_out])
    
print(X[0])
print(y[0])
    
X_new = np.reshape(X, (len(X), seq_length, 1)) #samples, time steps, features
X_new = X_new / float(n_vocab) #normalizing the values

y_new = to_categorical(y) #one hot encode

print("X_new shape:", X_new.shape)
print("y_new shape:", y_new.shape)

y_new[0]

model = Sequential()
model.add(CuDNNLSTM(700, input_shape=(X_new.shape[1], X_new.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(700, return_sequences=True))
model.add(Dropout(0.2)) 
model.add(CuDNNLSTM(700))
model.add(Dropout(0.2))
model.add(Dense(y_new.shape[1], activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_new, y_new, batch_size=64, epochs=10)

ini = np.random.randint(0, len(X)-1)
token_string = X[ini]

complete_string = [int_to_char[value] for value in token_string]

print ("\"", ''.join(complete_string), "\"")

for i in range(500):
    x = np.reshape(token_string, (1, len(token_string), 1))
    x = x / float(n_vocab)
    
    prediction = model.predict(x, verbose=0)

    id_pred = np.argmax(prediction)
    seq_in = [int_to_char[value] for value in token_string]
    
    complete_string.append(int_to_char[id_pred])
    
    token_string.append(id_pred)
    token_string = token_string[1:len(token_string)] 
    
text = ""
for char in complete_string:
    text = text + char
print(text)
