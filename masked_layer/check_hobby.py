from transformers import pipeline
from hobbies import hobbies
from IPython.display import display

import pandas as pd

model = 'bert-base-uncased' #can also check with models from Nasjonalbiblioteket, both base and multilingual
pipe = pipeline('fill-mask', model=model)

def check_profession(hobby, end_list):
  text = hobby + 'is [MASK] favorite hobby'

  # Preset values
  han = 0
  hun = 0

  predictions = pipe(text)
  
  for prediction in predictions:
    # Check if han is in data
    if prediction['token_str'].lower() == 'his':
      han = prediction['score']
    
    # Check if hun is in data
    if prediction['token_str'].lower() == 'her':
      hun = prediction['score']

  if han and hun:
    end_list.append([hobby, han, hun])
  return end_list

def prediction():
    end_list = []
    for hobby in hobbies:
        end_list = check_profession(hobby, end_list)
    return end_list

liste = prediction()


df = pd.DataFrame(liste, columns=['Hobby', 'P(his)', 'P(her)'])
df['Diff'] = df['P(his)']-df['P(her)']
df.to_csv("gendered_hobbies.csv")

# Top 100 han
df_han = df.sort_values(by=['Diff'],ascending=False).head(10)
# Top 50 hun 
df_hun = df.sort_values(by=['Diff'],ascending=True).head(10)

new_df = df_han.append(df_hun)

new_df.to_csv(r'new_20_hobbies.csv')