#!/usr/bin/env python
# coding: utf-8




import numpy as np
from keras.preprocessing.text import Tokenizer
import scipy.stats as sp
from collections import Counter
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from sklearn.utils import check_random_state
from itertools import chain
from sklearn.utils import check_random_state




# load in the data
list = []
with open('data.txt', 'r') as fin:
    data = fin.read().lower()
    
# select percentage of data, original data length = 2947366
data = data[:2947366]      #100% of the data
list = data.split()

# split data in training and test data
train, test = train_test_split(list, test_size = 0.2, random_state = 0)

# turn train and test set into numpy arrays.
array_train = np.array(train)
array_test = np.array(test)




# Fit the tokenizer on the training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([train])

# give every distinct word a distinct numeral value
index = 1
word_to_index = {}
for word in train:
    if word in word_to_index:
        # already seen
        continue
    word_to_index[word.lower()] = index
    index += 1

    
# create a list of only the integer values of the words
integer_list = []
for word in train:
    integer_list.append(word_to_index[word.lower()])
len(integer_list)
    
# get the maximum number from the integer list, which is equal to the size of the vocabulary  
vocab_size = max(integer_list)
print(vocab_size)
# transform integerlist into integer array
integer_array = np.array(integer_list)





# Create a dictionary containing the counts each word occurs in the dataset
counts_dict = Counter(integer_list)

#create a list consisting of only the counts.
counts_list = []
for value in counts_dict.values():
    counts_list.append(value)
     
    
# create a dictionary containing the probability of a word occurring. <- initial probs
frequencies = {key:float(value)/sum(counts_dict.values()) for (key,value) in counts_dict.items()}
frequency_list = []
for value in frequencies.values():
    frequency_list.append(value)





# create a list of bigrams
bigrams = []
for i in range(len(integer_list)):
    if i+1 in range(len(integer_list)):
        bigrams.append((integer_list[i], integer_list[i+1]))

# Create a dictionary containing the counts each bigram occurs in the dataset
bigrams_dict = Counter(bigrams)
values = bigrams_dict.values()


# create a dictionary containing the probability of a bigram occurring. <- trans_probs
bigrams_frequencies = {key:float(value)/sum(bigrams_dict.values()) for (key,value) in bigrams_dict.items()}
bigrams_frequency_list = []
for value in bigrams_frequencies.values():
    bigrams_frequency_list.append(value)
    #vocab_size
transitions = np.zeros((vocab_size, vocab_size), dtype=np.float)
for (state1, state2), probability in bigrams_frequencies.items():
    try:
        transitions[state1, state2] = probability
    except IndexError:
        print("-")
        




# Create the model and fit it with the starting probability and transition probabaility
model = hmm.GaussianHMM(n_components=vocab_size, covariance_type="full", verbose = 1)
model.start_prob_ = np.array(frequency_list)
model.transmat_ = np.array(transitions)
integer_array = integer_array.reshape(-1,1)

# Fit the model
model.fit(integer_array)




# https://github.com/hmmlearn/hmmlearn/issues/171
# function to transform from integer to word
output_i_w = []
def integer_to_word(val):
    val = round(val)
    for word, integer in word_to_index.items():
        if integer == val:
            output_i_w = word
        elif val > max_int:
            output_i_w = "OVERSHOOT"               #sometimes the predicted word is higher than the number of words so invalid
            break
        elif val < 0:
            output_i_w = "undershoot"              #sometimes the predicted word is negative, so invalid
            break
    return output_i_w

# function that gives integer value of word and returns it in a list.
output_w_i = []
def word_to_integer(valarray):
    for val in valarray:
        for word, integer in word_to_index.items():
            if word == val:
                return integer





#https://github.com/hmmlearn/hmmlearn/issues/171
def predict_state(valarray):
    states = model.predict(valarray)
    transmat_cdf = np.cumsum(model.transmat_, axis=1)
    random_state = check_random_state(model.random_state)
    next_state = (transmat_cdf[integer_list[-1]] > random_state.rand()).argmax()
    next_obs = model._generate_sample_from_state(next_state, random_state)
    print( next_obs)
    return next_obs




#  TESTING: 

# Fit the tokenizer on the test set.
tokenizer = Tokenizer()
tokenizer.fit_on_texts([test])

# give every distinct word a distinct numeral value
index = 1
word_to_index_test = {}
for word in test:
    if word in word_to_index_test:
        # already seen
        continue
    word_to_index_test[word.lower()] = index
    index += 1

    
# create a list of only the integer values of the words
integer_list_test = []
for word in test:
    integer_list_test.append(word_to_index_test[word.lower()])
max_int = max(integer_list_test)
print(max_int)
    


# transform integer list into integer numpy array
integer_array_test = np.array(integer_list_test)
integer_array_test = integer_array.reshape(-1,1)



# generate words looking only at the previous state
generated = []
def generation(data):
    for word in data:
        try:
            word = word_to_integer([word])
            generated_word = predict_state([[word]])
            generated_word = generated_word.tolist()
            generated.append(generated_word)
        except ValueError:
            generated.append([-1.11])  
        except IndexError:
            break
    return generated

generated = generation(test)

flat = [item for sublist in generated for item in sublist]


flat_integer_array_test =  [item for sublist in integer_array_test for item in sublist]
acc = []
for i in range(len(flat)):
    flat[i] = round(flat[i])
    try:
        if flat[i] == flat_integer_array_test[i+1]:
            acc.append('true')
        else: 
            acc.append('false')
    except IndexError:
        print('')

def accuracy_test(acc):
    truth = acc.count('true')
    accuracy = truth/len(acc)
    accuracy = accuracy * 100
    return accuracy
accuracy = accuracy_test(acc)
print("accuracy of the model is: ", accuracy)


# predict the probability of the test set
probsy = model.predict_proba(integer_array)               

# convert probabilties into entropy
entropy = sp.entropy(probsy)       

# calcuate average entropy
average_entropy = sum(entropy)/len(entropy)                
print('average entropy = ', average_entropy)

# do 2^(average_entropy) to get the perplexity
perplexity = 2**(average_entropy)                          
print('perplexity = ', perplexity)
