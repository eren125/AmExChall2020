# %%
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
import random
import json

# %%
# Extract the information from the intents.json file
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

# %%
# Build the model
model = keras.Sequential(
    [
        layers.Dense(8, name="dense1"),
        layers.Dense(8, name="dense2"),
        layers.Dense(len(output[0]), activation="softmax", name="output")
    ]) 
model.build(input_shape=[None,len(training[0])])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
# %%
# Training
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
epochs = 1000
history = model.fit(training, output, epochs=epochs, batch_size = 8)
model.save("model.keras")

# %%
# Plot the history of the accuracy gain during the training process
import matplotlib.pyplot as plt
L = history.history['accuracy']
plt.plot([i for i in range(len(L))], L)
plt.xlabel('epochs')
plt.ylabel('accuracy')

# %%
import numpy
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from tensorflow import keras
import random
import pickle
# %%
with open("data/intents.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)
# %%
training
# %%
output
# %%
from ai_bot import DenseNeuralNet
# %%
message = DenseNeuralNet("D0192RAS4MR", "<@U0192K8QJQJ> Hi!").get_message_payload()
# %%
from slack import WebClient
import os
slack_web_client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])
response = slack_web_client.chat_postMessage(**message)
# %%
