# %%
import pandas as pd
import json
import os

# %%
## Script to extract handmade questions into a json file for the training process
with open("data/new.json") as file:
    data = json.load(file)

for i in range(len(data["intents"][1:])):
    df = pd.read_excel("../data_amex/"+data["intents"][i+1]['tag'].strip().replace(" ","_")+".xlsx",header=None)
    if df.empty:
        L = []
    else:
        L = list(df[0])
    data['intents'][i+1]['patterns'] = L

# Save the dictionnary into a json file
with open('data/intents.json', "w") as f:
    json.dump(data, f, indent=4)

# %%
# # Data from csv file (recommandation of trip from Lonely Planet)
# df_in = pd.read_csv('AE Challenge.csv',sep=";")[['destination','tag','pattern','response/website']]

# # Rename and process the data
# df = df_in[['destination','tag']].copy()
# df['patterns'] = df_in['pattern'].str.split("/")
# df['responses'] = df_in['response/website'].str.split("\n")

# # Create a dictionnary
# L = [df[['tag','patterns','responses']].iloc[i].to_dict() for i in range(len(df))]
# df_dict = {'intents': data['intents'][:3]+L}
