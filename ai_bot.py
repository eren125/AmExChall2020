import numpy
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import random
import pickle
import json

class DenseNeuralNet:
    """Gives the appropriate answer to a quesiton asked to app_mention in the Channel the bot is subscribed"""
    """The chatbot uses a model bag of words for the questions encoding and Keras Dense Neural Network to 
       predict the intent and answer the question"""
    THRESHHOLD = 0.8
    def __init__(self, channel,inp):
        self.channel = channel
        self.username = "amexbot"
        self.icon_emoji = ":robot_face:"
        self.timestamp = ""
        self.inp = inp

    def get_message_payload(self,data,words,labels,model):
        return {
            "ts": self.timestamp,
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": [
                *self._predict(self.inp,data,words,labels,model) ## fuse predict with get_task_block
            ],
        }

    def _train(self,epochs=200,data_file="data/intents",model_file='model/model.keras'):
        
        with open(data_file+'.json') as file:
            data = json.load(file)
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

        with open(data_file+'.pickle', "wb") as f:
            pickle.dump((words, labels, training, output), f)

        model = keras.Sequential(
            [
                layers.Dense(128, activation='relu', name="dense1"),
                layers.Dropout(0.5),
                layers.Dense(64, activation='relu', name="dense2"),
                layers.Dropout(0.5),
                layers.Dense(len(output[0]), activation="softmax", name="output")
            ]) 
        model.build(input_shape=[None,len(training[0])])
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # Stochastic Gradient Descent
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        hist = model.fit(training, output, epochs=epochs, batch_size = 5, verbose=1)
        model.save(model_file, hist)
        return hist 
        
    def _load_data(self,data_file="data/new",model_file='model/model3.keras'):
        with open(data_file+'.json') as file:
            data = json.load(file)
        with open(data_file+'.pickle', "rb") as f:
            words, labels, training, output = pickle.load(f)
        model = keras.models.load_model(model_file)
        return(data,words,labels,model)

    def _predict(self,inp,data,words,labels,model):
        results = model.predict([self.bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if results[0][results_index]>self.THRESHHOLD:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    if tag in ['greeting','identity','goodbye']:
                        response=responses[0]
                    else:
                        response = ("I understand that you like %s. These are my recommandations: \n"%tag + "\n".join(responses))
            else: 
                response='Sorry, I didn\'t understand. Can you reformulate?'

        return self._get_task_block(response,tag)

    def bag_of_words(self,s, words):
        bag = [0 for _ in range(len(words))]

        s_words = word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1
                
        return bag

    @staticmethod
    def _get_task_block(text,information):
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": information}]},
        ]