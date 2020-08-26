# %%
import pandas as pd
import json

# %%
# Data from tutorial
with open("data/intents.json") as file:
    data = json.load(file)

# Data from csv file (recommandation of trip from Lonely Planet)
df_in = pd.read_csv('AE Challenge.csv',sep=";")[['destination','tag','pattern','response/website']]

# Rename and process the data
df = df_in[['destination','tag']].copy()
df['patterns'] = df_in['pattern'].str.split("/")
df['responses'] = df_in['response/website'].str.split("\n")

# Create a dictionnary
L = [df[['tag','patterns','responses']].iloc[i].to_dict() for i in range(len(df))]
df_dict = {'intents': data['intents'][:3]+L}

# Save the dictionnary into a json file
with open('data/intents.json', "w") as f:
    json.dump(df_dict, f, indent=4)

# %%
