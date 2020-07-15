#!/usr/bin/env python
# coding: utf-8

import nltk
from nltk.lm import MLE
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, defaultdict
from nltk import bigrams
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.lm import Vocabulary


# load in data
with open('data.txt', 'r') as fin:
    data = fin.read().lower()


# split the original data into individual sentences.
split_data = data.split("\n")


# Split the sentences into individual lists
split_data2 = []    
for sentence in split_data:
    split_data2.append(sentence.split())



# split the  data into a training and a test set
train, test = train_test_split(split_data, test_size = 0.2, random_state = 0)

# tokenize the train data
tokenized_text_train = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train]

# tokeneize the test data.
tokenized_text_test = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test]


# initialize the bigram model.

# make a list of bigrams in the test data
train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text_train]

# get all the individual words
words = [word for sent in tokenized_text_train for word in sent]

# pad the vocabulary
padded_vocab = Vocabulary(words)

# initialize and train the model
n = 2
model = MLE(n)
model.fit(train_data, padded_vocab)


# In[17]:


# make bigrams of the test data
test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text_test]
perplexity_list_bigram = []
for t in test_data:
    perplexity_list_bigram.append(model.perplexity(t))

#If a word is not in the training set, the model cannot make a prediction of that word, hence it will give an infinite perplexity value. we decided to remove these values.
perplexity_list_bigram = [x for x in perplexity_list_bigram if x != float('inf')]

# we take the average of the perplexities per bigrams.
average_perplexity_bigram = sum(perplexity_list_bigram) / len(perplexity_list_bigram)
print(average_perplexity_bigram)


# In[18]:


words = [word for sent in tokenized_text_test for word in sent]

#generate words from the bigrams and append them to generated
generated = []
def generation (data):
    for sentence in data:
        for word in sentence:
            try:
                generated_word = model.generate(text_seed = [word])
                generated.append(generated_word)
                break
            except ValueError:
                generated.append('VALUEERROR')
    return generated

# generate samples from the test set
generated = generation(tokenized_text_test)


# In[19]:


# Calculate the accuracy
acc = []
for i in range(len(generated)): 
    if generated[i] == words[i+1]:
        acc.append('true')
    else: 
        acc.append('false')
        
def accuracy_test(acc):
    truth = acc.count('true')
    accuracy = truth/len(acc)
    accuracy = accuracy * 100
    return accuracy

accuracy_test(acc)


# Trigrams


# split the trainginset into trigrams
train_data = [nltk.trigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text_train]
# take every word from the tokenized training set and pad the vocabulary.
words = [word for sent in tokenized_text_train for word in sent]
padded_vocab = Vocabulary(words)
# initialize and fit the model
n = 3
trigram = MLE(n)
trigram.fit(train_data, padded_vocab)



words = [word for sent in tokenized_text_test for word in sent]
#generate words from the bigrams and append them to generated
generated = []
def generation_trigram (data):
    for j in range(len(words)):
        try:
            generated_word = trigram.generate(text_seed = [words[j], words[j+1]])
            generated.append(generated_word)
        except ValueError:                                      
            generated.append('VALUEERROR')
        except IndexError:
            break
    return generated

generated_trigrams = generation_trigram([tokenized_text_test])



# Calculate accuracy for the trigrams
acc = []
for i in range(len(generated)):
    try: 
        if generated_trigrams[i] == words[i+2]:
            acc.append('true')
        else: 
            acc.append('false')
    except IndexError: 
        break
def accuracy_test_trigram(acc):
    truth = acc.count('true')
    accuracy = truth/len(acc)
    accuracy = accuracy * 100
    return accuracy

accuracy_test_trigram(acc)





# calucalte perplexity  for the trigrams
test_data = [nltk.trigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text_test]
perplexity_list_trigram = []
for t in test_data:
    perplexity_list_trigram.append(trigram.perplexity(t))
    
#If a word is not in the training set, the model cannot make a prediction of that word, hence it will give an infinite perplexity value. we decided to remove these values.
perplexity_list_trigram = [x for x in perplexity_list_trigram if x != float('inf')]
# take the average perplexity
average_perplexity_trigram = sum(perplexity_list_trigram) / len(perplexity_list_trigram)
    
print(average_perplexity_trigram)






