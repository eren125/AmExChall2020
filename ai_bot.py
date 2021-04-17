import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import re
import heapq
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras import Input,Model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Bidirectional, Flatten,GlobalAveragePooling1D,Lambda
from keras.callbacks import ModelCheckpoint

from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import bert
import random
import pickle
import json

MAX_NB_WORDS = 500
MAX_SEQUENCE_LENGTH = 250

class LoadingData():
    """Load data into a pandas DataFrame and preprocess it"""
    def __init__(self,verbose=0):
        filename = 'data/intents.json'
        json_file = json.load(open(filename))
        # Explode the list of questions of the dataframe
        self.data_frame = pd.DataFrame(json_file['intents'])[['tag','patterns']].explode('patterns')
        # Tensorized input (series of questions)
        patterns_series = self.data_frame['patterns'].apply(self._clean_text)
        self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        self.tokenizer.fit_on_texts(patterns_series.values)
        # self.X_ = self._tensorize(patterns_series,verbose=verbose)
        self.X = np.array(patterns_series)
        # Tensorized label using dummy vectors (1 if is label else 0)
        self.Y = pd.get_dummies(self.data_frame['tag']).values
        if verbose==1:
            print('Shape of label tensor:', self.Y.shape)
        # Decoding an output vector into an intent label and then into a response TODO: make a method 
        Y_to_label = np.sort(self.data_frame['tag'].unique())
        self.cat_to_tag = dict(enumerate(Y_to_label))
        self.tag_to_cat = {value:key for (key,value) in self.cat_to_tag.items()}
        df_tag_to_response = pd.DataFrame(json_file['intents'])[['tag','responses']]
        self.tag_to_response = dict(df_tag_to_response.set_index('tag')['responses'])
        # Train - Test split
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X,self.Y, test_size = 0.18, random_state = 42)

    def _clean_text(self,text):
        """Returns a lemmatized and cleaned text from raw data"""
        REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))

        text = text.lower() # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
        text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

        text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
        return text

    def _tensorize(self,patterns_series,verbose=1):
        """Returns a 2D array from an iterable of text"""
        X = self.tokenizer.texts_to_sequences(patterns_series.values)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        if verbose ==1:
            print('Shape of data tensor:', X.shape)
        return np.array(X)

class ModelFcnn():
    """Fully connected Neural Network training methods"""
    def __init__(self):
        self.model = None
        self._type = "FCNN"
        self.VOCAB = {}

    def _bagging(self, corpus):
        sentence_vectors = []
        for sentence in corpus:
            sentence_tokens = word_tokenize(sentence)
            sent_vec = []
            for token in self.VOCAB:
                if token in sentence_tokens:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            sentence_vectors.append(sent_vec)
        return np.array(sentence_vectors)
           
    def get_input_array(self,s,_clean_text,_tensorize):
        corpus = np.array([_clean_text(sentence) for sentence in s])
        return self._bagging(corpus)

    def build(self,X,Y, _clean_text):
        wordfreq = {}
        corpus = np.array([_clean_text(sentence) for sentence in X])
        for sentence in corpus:
            tokens = word_tokenize(sentence)
            for token in tokens:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1
        self.VOCAB = heapq.nlargest(MAX_NB_WORDS, wordfreq, key=wordfreq.get)
        
        self.model = Sequential()
        self.model.add(Input(shape=(MAX_NB_WORDS,)))
        self.model.add(Dense(512,activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256,activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128,activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64,activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(Y.shape[1], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self,X,Y,save_file='model/FCNN.h5',epochs = 100,batch_size = 64,validation_split=0.2):
        checkpoint = ModelCheckpoint(save_file, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split,callbacks=[checkpoint])    

class ModelRnnlm():
    """Fully connected Neural Network training methods"""
    def __init__(self):
        self.model = None
        self._type = "RNNLM"
        self.hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1", output_shape=[50],
                           input_shape=[], dtype=tf.string,trainable=True)
    
    def get_input_array(self,s,_clean_text,_tensorize):
        return np.array([_clean_text(sentence) for sentence in s])

    def _build(self,X_train,Y_train):
        self.model = Sequential()
        self.model.add(self.hub_layer)
        self.model.add(Dropout(0.5))
        self.model.add(Dense(Y_train.shape[1], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def _train(self,X_train,Y_train,save_file='model/RNNLM.h5',epochs = 100,batch_size = 64,validation_split=0.2):
        checkpoint = ModelCheckpoint(save_file, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')

        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split,callbacks=[checkpoint])    

class ModelLstm():
    def __init__(self):
        self.model = None
        self._type = "LSTM"
    
    def get_input_array(self,s,_clean_text,_tensorize):
        sentences = np.array([_clean_text(sentence) for sentence in s])
        inp = _tensorize(pd.Series(sentences),verbose=0)
        return inp
        
    def _build(self,X_train,Y_train,EMBEDDING_DIM = 128):
        self.model = Sequential()
        self.model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(Y_train.shape[1], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def _train(self,X_train,Y_train,save_file='model/LSTM.h5',epochs = 100,batch_size = 64,validation_split=0.2):
        checkpoint = ModelCheckpoint(save_file, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')

        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=validation_split,callbacks=[checkpoint])

class ModelBiLstm():
    def __init__(self):
        self.model = None
        self._type = "BiLSTM"

    def get_input_array(self,s,_clean_text,_tensorize):
        sentences = np.array([_clean_text(sentence) for sentence in s])
        inp = _tensorize(pd.Series(sentences),verbose=0)
        return inp

    def _build(self,X_train,Y_train,EMBEDDING_DIM = 128):
        self.model = Sequential()
        self.model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(Bidirectional(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(Dense(Y_train.shape[1], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def _train(self,X_train,Y_train,save_file='model/BiLSTM.h5',epochs = 100,batch_size = 64,validation_split=0.2):
        checkpoint = ModelCheckpoint(save_file, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')

        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=validation_split,callbacks=[checkpoint])

class ModelBiLstm2():
    def __init__(self):
        self.model = None
        self._type = "2BiLSTM"

    def get_input_array(self,s,_clean_text,_tensorize):
        sentences = np.array([_clean_text(sentence) for sentence in s])
        inp = _tensorize(pd.Series(sentences),verbose=0)
        return inp

    def _build(self,X_train,Y_train,EMBEDDING_DIM = 100):
        self.model = Sequential()
        self.model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(Bidirectional(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)))
        self.model.add(Bidirectional(LSTM(EMBEDDING_DIM//2, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(Dense(Y_train.shape[1], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def _train(self,X_train,Y_train,save_file='model/BiLSTM2.h5',epochs = 100,batch_size = 64,validation_split=0.2):
        checkpoint = ModelCheckpoint(save_file, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')

        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=validation_split,callbacks=[checkpoint])

class ModelBert():
    def __init__(self,in_len,out_len,save_file="model/Bert.h5",trainable=True):
        self.model = None
        self._type = "BERT"
        bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
        self.bert_module = hub.KerasLayer(bert_path,trainable=trainable)
        FullTokenizer=bert.bert_tokenization.FullTokenizer
        vocab_file = self.bert_module.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_module.resolved_object.do_lower_case.numpy()
        self.tokenizer = FullTokenizer(vocab_file,do_lower_case)
        self.out_len = out_len
        self.in_len = in_len
        self.save_file = save_file

    def get_masks(self,tokens, max_seq_length):
        mask_data =  [1]*len(tokens) + [0] * (max_seq_length - len(tokens))
        return mask_data

    def get_segments(self,tokens, max_seq_length):
        '''
        Segments: 0 for the first sequence, 
        1 for the second
        '''
        segments = []
        segment_id = 0
        for token in tokens:
            segments.append(segment_id)
            if token == "[SEP]":
                segment_id = 1
        '''Remaining are padded with 0'''
        remaining_segment = [0] * (max_seq_length - len(tokens))
        segment_data = segments + remaining_segment
        return segment_data
    
    def get_ids(self,tokens, tokenizer, max_seq_length):
        token_ids = tokenizer.convert_tokens_to_ids(tokens,)
        remaining_ids = [0] * (max_seq_length-len(token_ids))
        input_ids = token_ids + remaining_ids
        return input_ids
    
    def get_input_data(self,sentence,max_len):
        sent_token = self.tokenizer.tokenize(sentence)
        sent_token = sent_token[:max_len]
        sent_token = ["[CLS]"] + sent_token + ["[SEP]"]

        id = self.get_ids(sent_token, self.tokenizer, self.in_len )
        mask = self.get_masks(sent_token, self.in_len )
        segment = self.get_segments(sent_token, self.in_len )
        input_data = [id,mask,segment]
        return input_data

    def get_input_array(self,sentences,_clean_text,_tensorize,verbose=0):
        input_ids, input_masks, input_segments = [], [], []
        if verbose==0:
            sentences_ = [_clean_text(w) for w in sentences]
        else:
            sentences_ = tqdm([_clean_text(w) for w in sentences],position=0, leave=True)
        for sentence in sentences_:
            ids,masks,segments=self.get_input_data(sentence,self.in_len-2)

            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
            
        input_array = [np.asarray(input_ids, dtype=np.int32),np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]
        return input_array

    def _build(self): 
        in_id = Input(shape=(self.in_len,), dtype=tf.int32, name="input_ids")
        in_mask = Input(shape=(self.in_len,), dtype=tf.int32, name="input_masks")
        in_segment = Input(shape=(self.in_len,), dtype=tf.int32, name="segment_ids")
        
        bert_inputs = [in_id, in_mask, in_segment]
        bert_pooling_out, bert_sequence_out = self.bert_module(bert_inputs)
        
        out = GlobalAveragePooling1D()(bert_sequence_out)
        out = Dropout(0.2)(out)
        out = Dense(128, activation="tanh", name="hidden")(out)
        out = Dropout(0.2)(out)
        out = Dense(self.out_len, activation="softmax", name="dense_output")(out)
        self.model = Model(inputs=bert_inputs, outputs=out)
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

    def _train(self,X,Y,batch_size,epochs,validation_split=0.2):
        print("Fitting to model")
        checkpoint = ModelCheckpoint(self.save_file, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')
        self.model.fit(X,Y,epochs=epochs,batch_size=batch_size,
                       validation_split=validation_split,shuffle=True,callbacks=[checkpoint])
        print("Model Training complete.")

class SlackMessage():
    """Gives the appropriate answer to a question asked to app_mention in the Channel the bot is subscribed
       The chatbot uses a model bag of words for the questions encoding and Keras Dense Neural Network to 
       predict the intent and answer the question"""
    THRESHOLD = 0.5
    def __init__(self, channel,inp):
        self.channel = channel
        self.username = "amexbot"
        self.icon_emoji = ":robot_face:"
        self.timestamp = ""
        self.inp = inp

    def get_message_payload(self,model_objects,_clean_text,_tensorize,cat_to_tag,tag_to_response):
        text, information = self._predict(self.inp,model_objects,_clean_text,_tensorize,cat_to_tag,tag_to_response)
        return {
            "ts": self.timestamp,
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": [
                *self._get_task_block(text,information)
            ],
        }

    def _predict(self,s,model_objects,_clean_text,_tensorize,cat_to_tag,tag_to_response):
        N = len(model_objects)
        information = ""
        for i in range(N):
            input_array = model_objects[i].get_input_array([s],_clean_text,_tensorize)
            results_1 = model_objects[i].model.predict(input_array)

            results_index_1 = np.argmax(results_1[0])
            percentage_1 = results_1[0][results_index_1]*100

            results_2 = [x if x*100<percentage_1 else 0 for x in results_1[0]]
            results_2[3] = 0 #greeting set to zero > tag_to_cat['greeting']
            results_index_2 = np.argmax(results_2)
            percentage_2 = results_2[results_index_2]*100

            pred_1 = cat_to_tag[results_index_1]
            pred_2 = cat_to_tag[results_index_2]

            information += "%s: %s(%.2f%%), %s(%.2f%%)\n"%(model_objects[i]._type,pred_1,percentage_1,pred_2,percentage_2)
            if i==N-1:
                if percentage_1>75: # in %
                    responses = tag_to_response[pred_1]
                    if pred_1 in ['greeting']:
                        response=responses[0]
                    else:
                        response = ("I understand that you like %s. These are my recommandations: \n"%pred_1 + "\n".join(responses))
                
                elif percentage_1>30: # in %
                    responses = tag_to_response[pred_1]
                    responses_2 = tag_to_response[pred_2]
                    response = ("I understand that you like %s and %s. These are my recommandations: \n"%(pred_1,pred_2) + "\n".join(responses+responses_2))
                
                else: 
                    response='Sorry, I didn\'t understand. Can you reformulate?'

        return (response,information)

    @staticmethod
    def _get_task_block(text,information):
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": information}]},
        ]

class OnboardingMessage():
    """Constructs the onboarding message and stores the state of which tasks were completed."""
    # https://github.com/slackapi/python-slackclient/issues/392
    # https://github.com/slackapi/python-slackclient/pull/400
    WELCOME_BLOCK = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                "Welcome to AmexBot's Slack Channel! :wave: The whole team including me are so glad you're here. :blush:\n\n"
                "I am an experimental chatting robot that has been specifically trained for giving travel recommendations\n\n"
                "Please ask your question using the app_mention: @amexbot.\n\n"
                "Try to ask questions on a trip in Beijing that are related to these 7 classes:\n\n"
                "\t• historical monuments\n \t• food\n \t• animals(zoo) \n \t• cruise \n \t• night tour \n \t• relaxation\n \t• show and concerts\n"
                "You can also greet me (this is the 8th label I can predict)\n"
                "*Get started by completing the steps below:*"
            ),
        },
    }
    DIVIDER_BLOCK = {"type": "divider"}

    def __init__(self, channel):
        self.channel = channel
        self.username = "amexbot"
        self.icon_emoji = ":robot_face:"
        self.timestamp = ""
        self.reaction_task_completed = False
        self.pin_task_completed = False

    def get_message_payload(self):
        return {
            "ts": self.timestamp,
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": [
                self.WELCOME_BLOCK,
                self.DIVIDER_BLOCK,
                *self._get_reaction_block(),
                self.DIVIDER_BLOCK,
                *self._get_pin_block(),
            ],
        }

    def _get_reaction_block(self):
        task_checkmark = self._get_checkmark(self.reaction_task_completed)
        text = (
            f"{task_checkmark} *Add an emoji reaction to this message* :thinking_face:\n"
            "Indicate if you liked the amexbot experience with a well-chosen emoji :slightly_smiling_face:"
        )
        information = (
            ":thumbsup: / :thumbsdown:"
        )
        return self._get_task_block(text, information)

    def _get_pin_block(self):
        task_checkmark = self._get_checkmark(self.pin_task_completed)
        text = (
            f"{task_checkmark} *Pin this message* :round_pushpin:\n"
            "Important messages and files can be pinned to the details pane in any channel or"
            " direct message, including group messages, for easy reference."
        )
        information = (
            ":information_source: *<https://get.slack.help/hc/en-us/articles/205239997-Pinning-messages-and-files"
            "|Learn How to Pin a Message>*"
        )
        return self._get_task_block(text, information)

    @staticmethod
    def _get_checkmark(task_completed: bool) -> str:
        if task_completed:
            return ":white_check_mark:"
        return ":white_large_square:"

    @staticmethod
    def _get_task_block(text, information):
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": information}]},
        ]