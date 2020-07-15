#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import numpy as np
from numpy import array
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from sklearn.model_selection import train_test_split


# In[2]:


# load in data
with open('D:\data_goed.txt', 'r') as fin:
    data = fin.read().lower()
data = data
split_data = data.split("\n")

split_data2 = []    
for sentence in split_data:
    split_data2.append(sentence.split())

# Split data into training and test set
train, test = train_test_split(split_data2, test_size = 0.2, random_state = 0)



# In[3]:


# fit the Tokenizer on the training data.
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# encode the corpus into integers
input_seq_list = []
for line in train:
    encoded_line = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded_line)):
        sequence = encoded_line[:i+1]
        input_seq_list.append(sequence)


# In[4]:


# find the longest input sequence in the data, and then pad the sentences that were prepared based on that length
max_length = max([len(seq) for seq in input_seq_list])
input_seq_list = pad_sequences(input_seq_list, maxlen=max_length, padding='pre')

# determine the size of the vocabulary
vocab_size = len(tokenizer.word_index) + 1


# In[5]:


# split the input_seq_list into input elements X and output elements y
input_seq_list = np.array(input_seq_list)
X = input_seq_list[:,:-1]
y = input_seq_list[:,-1]

# Use one-hot encoding on y
y = to_categorical(y, num_classes=vocab_size)


# In[6]:


# define the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_length-1))

# number of hidden units
model.add(LSTM(50)) 
model.add(Dense(vocab_size, activation='softmax'))

# compile network with categoritcal crossentropy as loss, adam as optimizer and accuracy aas measure
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network for 15 epochs
model.fit(X, y, epochs=15, verbose=2)


# <b> Generating Text:
#   

# In[7]:


# from: https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/

def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded_line = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded_line = pad_sequences([encoded_line], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded_line, verbose=0)
        print(yhat)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text


# <B>Evaluation on the test set: 

# In[22]:


# encode the sequences of the test set into integers.
input_seq_list_test = []
for line in test:
    encoded_line = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded_line)):
        sequence = encoded_line[:i+1]
        input_seq_list_test.append(sequence)




# pad input sequences to the length of the longest sequece
max_length = max([len(seq) for seq in sequences])
input_seq_list_test = pad_sequences(input_seq_list_test, maxlen=max_length, padding='pre')
vocab_size_test = len(tokenizer.word_index) + 1

# split the input_seq_list into input elements X and output elements y
input_seq_list_test = np.array(input_seq_list_test)
X_test, y_test = input_seq_list_test[:,:-1],input_seq_list_test[:,-1]

# use one hot encoding on y
y_test = to_categorical(y_test, num_classes=vocab_size_test)


# In[23]:


# evaluate the model, Note that this gives cross-entropy as loss, so we still have to take the exponent to acquire perplexity. 
results = model.evaluate(X_test, y_test)
print('test loss, test acc:', results)


# In[ ]:




