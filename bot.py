import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras


import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import pandas as pd
import pickle
import random
#nltk.download('punkt')

with open('intents2.json') as file:
    data = json.load(file)
    
ws = []
cs = []
ds = []
ignore_ws = ['?']
# Check all intents
for i in data['intents']:
    # Check for pattern in each intent
    for p in i['patterns']:
        # get token from all words
        w = nltk.word_tokenize(p)
        # add to our word list
        ws.extend(w)
        ds.append((w, i['tag']))
        # add to class list
        if i['tag'] not in cs:
            cs.append(i['tag'])
# remove duplicates
ws = [stemmer.stem(w.lower()) for w in ws if w not in ignore_ws]
ws = sorted(list(set(ws)))
# sort classes
cs = sorted(list(set(cs)))
# Docs = combination between patterns and intents
print (len(ds), "docs")
# classes = intents
print (len(cs), "classes", cs)
# words = all ws, vocabulary
print (len(ws), "unique words", ws)


# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(cs)
# training set, bag of words for each sentence
for doc in ds:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word - create base word, in attempt to represent related words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in ws:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[cs.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Fit the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
    
def predict_class(sentence):
    ERROR_THRESHOLD = 0.25
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, ws)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((cs[r[0]], str(r[1])))
    # return tuple of intent and probability
    
    return return_list
	
def getResponse(ints, intents_json):
    tag = ints[0][0]
    print(tag)
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text)
    res = getResponse(ints,data)
    return res