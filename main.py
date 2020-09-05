import ai_bot

# Load model once and for all (using load data function)
data = ai_bot.LoadingData()
cat_to_tag,tag_to_response = data.cat_to_tag,data.tag_to_response
_clean_text,_tensorize = data._clean_text,data._tensorize

X_shape,Y_shape = data.X,data.Y # Need data for the shaped to be used

model_obj_1 = ai_bot.ModelFcnn()
X,Y = model_obj_1.get_input_array(X_shape,_clean_text,_tensorize),Y_shape
model_obj_1._build(X,Y)
model_obj_1.model.load_weights("model/FCNN.h5")

model_obj_2 = ai_bot.ModelLstm()
X,Y = model_obj_2.get_input_array(X_shape,_clean_text,_tensorize),Y_shape
model_obj_2._build(X,Y,EMBEDDING_DIM = 128)
model_obj_2.model.load_weights("model/LSTM.h5")

model_obj_3 = ai_bot.ModelRnnlm()
X,Y = model_obj_3.get_input_array(X_shape,_clean_text,_tensorize),Y_shape
model_obj_3._build(X,Y)
model_obj_3.model.load_weights("model/RNNLM.h5")

model_obj_4 = ai_bot.ModelBert(512,Y_shape.shape[1])
model_obj_4._build()
model_obj_4.model.load_weights("model/Bert.h5")

model_objects = [model_obj_1,model_obj_2,model_obj_3,model_obj_4]

messenger = ai_bot.SlackMessage('','')

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        text, information = messenger._predict(inp,model_objects,_clean_text,_tensorize,cat_to_tag,tag_to_response)

        print(text)
        print(information)

chat()