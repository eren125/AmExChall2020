import numpy
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from tensorflow import keras
from tensorflow.keras import layers
import random
import json
import pickle

with open("data/intents.json") as file:
    data = json.load(file)
try:
    with open("data/intents.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words, labels, docs_x, docs_y = [], [], [], []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training, output = [], []

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

    with open("data/intents.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

try:
    model = keras.models.load_model('model/model.keras')
except:
    model = keras.Sequential(
        [
            layers.Dense(8, name="dense1"),
            layers.Dense(8, name="dense2"),
            layers.Dense(len(output[0]), activation="softmax", name="output")
        ]) 
    model.build(input_shape=[None,len(training[0])])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    epochs=1000
    model.fit(training, output, epochs=epochs, batch_size = 8)
    model.save("model/model.keras")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return bag

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
