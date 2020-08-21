import numpy
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from tensorflow import keras
import random
import pickle
import json

class DenseNeuralNet:
    """Gives the appropriate answer to a quesiton asked to app_mention in the Channel the bot is subscribed"""
    
    def __init__(self, channel,inp):
        self.channel = channel
        self.username = "amexbot"
        self.icon_emoji = ":robot_face:"
        self.timestamp = ""
        self.inp = inp

    def get_message_payload(self):
        return {
            "ts": self.timestamp,
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": [
                *self._predict(self.inp) ## fuse predict with get_task_block
            ],
        }

    def _predict(self,inp):
        with open("data/intents.json") as file:
            data = json.load(file)
        with open("data/intents.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
        model = keras.models.load_model('model/model.keras')
        
        results = model.predict([self.bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        response = (f"{random.choice(responses)}")
        return self._get_task_block(response)

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
    def _get_task_block(text):
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        ]