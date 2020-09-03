# %%
import ai_bot
import matplotlib.pyplot as plt

data = ai_bot.LoadingData() # A class containing preprocessed data

# %%
# Full bag-of-words / FCNN does not work (accuracy low even with a high dimensionality 430k params)
Model = ai_bot.ModelFcnn()
Model._build(data.X_,data.Y)
Model._train(data.X_,data.Y,epochs = 100,batch_size = 8,validation_split=0.1)

# %%
# A model using a pretrained RNN model with a simple FClayer for the classification
Model = ai_bot.ModelRnnlm()
X_train, Y_train = data.X_train,data.Y_train # text data (train)
Model._build(X_train,Y_train)
Model._train(X_train,Y_train,epochs = 20,batch_size = 8,validation_split=0.15)
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
X_test, Y_test = data.X_test,data.Y_test # text data (test)
loss, accuracy = Model.model.evaluate(X_test,Y_test)

# %%
# Long-short term memory model with embedding
Model = ai_bot.ModelLstm()
X_train, Y_train = data.X_train_,data.Y_train_ # vectorized data (train)
Model._build(X_train,Y_train,EMBEDDING_DIM = 128)
Model._train(X_train,Y_train,epochs = 20,batch_size = 8,validation_split=0.15)
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
X_test, Y_test = data.X_test_,data.Y_test_ # vectorized data (test)
loss, accuracy = Model.model.evaluate(X_test,Y_test)

# %%
# Long-short term memory model with embedding and bidirectionality
Model = ai_bot.ModelBiLstm()
X_train, Y_train = data.X_train_,data.Y_train_ # vectorized data (train)
Model._build(data.X_,data.Y,EMBEDDING_DIM = 128)
Model._train(data.X_,data.Y,epochs = 20,batch_size = 8,validation_split=0.15)
# %%
history = Model.model.history.history
fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle("Training procedure for Bidirectional LSTM model")
ax1.set_title('Loss')
ax1.plot(history['loss'], label='train')
ax1.plot(history['val_loss'], label='validation')
ax1.legend()
ax2.set_title('Accuracy')
ax2.plot(history['accuracy'], label='train')
ax2.plot(history['val_accuracy'], label='validation')
ax2.legend()
# %%
X_test, Y_test = data.X_test_,data.Y_test_ # vectorized data (test)
loss, accuracy = Model.model.evaluate(X_test,Y_test)

# %%
# Copy the overall architecture of ELMo model (embedding/2BiLSTM/classification FCNN layer)
Model = ai_bot.ModelBiLstm2()
X_train, Y_train = data.X_train_,data.Y_train_ # vectorized data (train)
Model._build(data.X_,data.Y,EMBEDDING_DIM = 100)
Model._train(data.X_,data.Y,epochs = 20,batch_size = 8,validation_split=0.15)
# %%
history = Model.model.history.history
fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle("Training procedure for double layers of Bidirectional LSTM model")
ax1.set_title('Loss')
ax1.plot(history['loss'], label='train')
ax1.plot(history['val_loss'], label='validation')
ax1.legend()
ax2.set_title('Accuracy')
ax2.plot(history['accuracy'], label='train')
ax2.plot(history['val_accuracy'], label='validation')
ax2.legend()
# %%
X_test, Y_test = data.X_test_,data.Y_test_ # vectorized data (test)
loss, accuracy = Model.model.evaluate(X_test,Y_test)

# %%
# BERT training to fine-tune the pre-trained parameters
X_train, Y_train = data.X_train,data.Y_train # text data (train)
Model = ai_bot.ModelBert(len(X_train),Y_train.shape[1])
Model.bert_model()
Model._train(X_train, Y_train,epochs = 7,batch_size = 8,validation_split=0.15)
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
X_test, Y_test = data.X_test,data.Y_test # text data (test)
loss, accuracy = Model.model.evaluate(X_test,Y_test)

# %%
