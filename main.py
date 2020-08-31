# %%
import ai_bot
import matplotlib.pyplot as plt
MAX_NB_WORDS=128
data = ai_bot.LoadingData(verbose=1,MAX_NB_WORDS=MAX_NB_WORDS)

model_obj = ai_bot.ModelBert(MAX_NB_WORDS,len(data.cat_to_tag))

model_obj.bert_model()

model_obj.model_train(data.data_frame['patterns'],data.data_frame['tag'].apply(lambda x: data.tag_to_cat[x]),batch_size=5,num_epoch=200)

model_obj.model.save('model/Bert.h5')

# %%
data = ai_bot.LoadingData(verbose=1,MAX_NB_WORDS=MAX_NB_WORDS)

history = ai_bot.ModelBiLstm()._train(data.X_train,data.Y_train,batch_size=5,epochs=200,MAX_NB_WORDS=MAX_NB_WORDS)

# %%
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

# %%
try:
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()
    plt.show()
except:
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.legend()
    plt.show()

# %%
model = ai_bot.load_model('model/biLSTM.h5')
accr = model.evaluate(data.X_test,data.Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))