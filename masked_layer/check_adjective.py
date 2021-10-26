from transformers import pipeline
from process_adjective import adjectives
from IPython.display import display

import pandas as pd

model = 'ltgoslo/norbert' #can also check with models from Nasjonalbiblioteket, both base and multilingual
pipe = pipeline('fill-mask', model=model)

def check_adjective(adjective, end_list):
  text = '[MASK] er ' + adjective

  # Preset values
  han = 0
  hun = 0

  predictions = pipe(text)
  
  for prediction in predictions:
    # Check if han is in data
    if prediction['token_str'].lower() == 'han':
      han = prediction['score']
    
    # Check if hun is in data
    if prediction['token_str'].lower() == 'hun':
      hun = prediction['score']

  if han and hun:
    end_list.append([adjective, han, hun])
  return end_list

def prediction():
    end_list = []
    for adjective in adjectives:
        end_list = check_adjective(adjective, end_list)
    return end_list

liste = prediction()


df = pd.DataFrame(liste, columns=['Adjektiv', 'P(han)', 'P(hun)'])
df['Differanse'] = df['P(han)']-df['P(hun)']
df.to_pickle("gendered_adjectives.pkl")
display(df)
