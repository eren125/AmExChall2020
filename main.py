# %%
import ai_bot
data = ai_bot.LoadingData()
Model = ai_bot.ModelRnnlm()
Model._train(data.X,data.Y,epochs = 10,batch_size = 32)
Model.model.save_weights("model/RNNLM.h5")
# Model.model.load_weights("model/RNNLM.h5")
# %%
import ai_bot
data = ai_bot.LoadingData()
Model = ai_bot.ModelLstm()
Model._train(data.X_,data.Y,epochs = 10,batch_size = 32)
Model.model.save_weights("model/LSTM.h5")
# Model.model.load_weights("model/LSTM.h5")
# %%
import ai_bot
data = ai_bot.LoadingData()
Model = ai_bot.ModelBiLstm()
Model._train(data.X_,data.Y,epochs = 10,batch_size = 32)
Model.model.save_weights("model/BiLSTM.h5")
# Model.model.load_weights("model/BiLSTM.h5")
# %%
# BERT training session
import ai_bot
data = ai_bot.LoadingData()
Model = ai_bot.ModelBert(len(data.X),len(data.Y))
Model.bert_model()
Model._train(data.X,data.data_frame['tag'].apply(lambda x: data.tag_to_cat[x]),epochs = 10,batch_size = 32)
# Model.model.load_weights("model/Bert.h5")
# %%
