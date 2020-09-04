# %%
import ai_bot
import matplotlib.pyplot as plt

data = ai_bot.LoadingData() # A class containing preprocessed data
X_train, Y_train = data.X_train,data.Y_train # text data (train)
X_test, Y_test = data.X_test,data.Y_test # text data (test)
_clean_text,_tensorize = data._clean_text,data._tensorize
print("Training set's size: %s \nTest set's size: %s"%(len(X_train),len(X_test)))
print(data.cat_to_tag,data.tag_to_response)
# %%
# Full bag-of-words / FCNN does not work (accuracy low even with a high dimensionality 430k params)
Model = ai_bot.ModelFcnn()
# Preprocess data into bag-of-words
X_train_,Y_train_ = Model.get_input_array(X_train,_clean_text,_tensorize),Y_train # bag-of-words (train)
Model._build(X_train_,Y_train_)
Model._train(X_train_,Y_train_,epochs = 20,batch_size = 8,validation_split=0.1)
# %%
history = Model.model.history.history
fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle("Training procedure for LSTM model")
ax1.set_title('Loss')
ax1.plot(history['loss'], label='train')
ax1.plot(history['val_loss'], label='validation')
ax1.legend()
ax2.set_title('Accuracy')
ax2.plot(history['accuracy'], label='train')
ax2.plot(history['val_accuracy'], label='validation')
ax2.legend()
# %%
X_test_, Y_test_ = Model.get_input_array(X_test,_clean_text,_tensorize),Y_test # text data (test)
loss, accuracy = Model.model.evaluate(X_test_, Y_test_)
print("Test accuracy: %s"%accuracy)

# %%
# Long-short term memory model with embedding
Model = ai_bot.ModelLstm()
# Preprocess data into sequences
X_train_,Y_train_ = Model.get_input_array(X_train,_clean_text,_tensorize),Y_train # Preprocess data so that it corresponds to the model
Model._build(X_train_,Y_train_,EMBEDDING_DIM = 128)
Model._train(X_train_,Y_train_,epochs = 20,batch_size = 8,validation_split=0.15)
# %%
history = Model.model.history.history
fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle("Training procedure for LSTM model")
ax1.set_title('Loss')
ax1.plot(history['loss'], label='train')
ax1.plot(history['val_loss'], label='validation')
ax1.legend()
ax2.set_title('Accuracy')
ax2.plot(history['accuracy'], label='train')
ax2.plot(history['val_accuracy'], label='validation')
ax2.legend()
# %%
X_test_, Y_test_ = Model.get_input_array(X_test,_clean_text,_tensorize),Y_test # text data (test)
loss, accuracy = Model.model.evaluate(X_test_, Y_test_)
print("Test accuracy: %s"%accuracy)

# %%
# A model using a pretrained RNN model with a simple FClayer for the classification
Model = ai_bot.ModelRnnlm()
# Preprocess sentences into cleaned sentences
X_train_,Y_train_ = Model.get_input_array(X_train,_clean_text,_tensorize),Y_train # cleant sentences (train)
Model._build(X_train_,Y_train_)
Model._train(X_train_,Y_train_,epochs = 20,batch_size = 8,validation_split=0.15)
# %%
history = Model.model.history.history
fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle("Training procedure for RNNLM")
ax1.set_title('Loss')
ax1.plot(history['loss'], label='train')
ax1.plot(history['val_loss'], label='validation')
ax1.legend()
ax2.set_title('Accuracy')
ax2.plot(history['accuracy'], label='train')
ax2.plot(history['val_accuracy'], label='validation')
ax2.legend()
# %%
X_test_, Y_test_ = Model.get_input_array(X_test,_clean_text,_tensorize),Y_test # text data (test)
loss, accuracy = Model.model.evaluate(X_test_, Y_test_)
print("Test accuracy: %s"%accuracy)

# %%
# BERT training to fine-tune the pre-trained parameters
Model = ai_bot.ModelBert(512,Y_train.shape[1])
# Preprocess data into sequences
X_train_,Y_train_ = Model.get_input_array(X_train,_clean_text,_tensorize),Y_train # Preprocess data so that it corresponds to the model
Model.bert_model()
Model._train(X_train_,Y_train_,epochs = 7,batch_size = 8,validation_split=0.15)
# %%
history = Model.model.history.history
fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle("Training procedure for the BERT model")
ax1.set_title('Loss')
ax1.plot(history['loss'], label='train')
ax1.plot(history['val_loss'], label='validation')
ax1.legend()
ax2.set_title('Accuracy')
ax2.plot(history['accuracy'], label='train')
ax2.plot(history['val_accuracy'], label='validation')
ax2.legend()
# %%
X_test_, Y_test_ = Model.get_input_array(X_test,_clean_text,_tensorize),Y_test # text data (test)
loss, accuracy = Model.model.evaluate(X_test_, Y_test_)
print("Test accuracy: %s"%accuracy)

# %%
test_acc = [11.18,92.55,93.17,96.89]
model = ['FCNN','LSTM','RNNLM','BERT']
plt.scatter(model,test_acc,marker='x')
plt.xlabel("Deep learning model")
plt.ylabel("Test accuracy in %")
plt.ylim(bottom=0,top=100)
plt.title("Test accuracy comparison between the studied DL models")