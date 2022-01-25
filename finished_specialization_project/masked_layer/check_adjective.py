from transformers import pipeline
from process_adjective import adjectives
from IPython.display import display

import pandas as pd

model = 'ltgoslo/norbert' #can also check with models from Nasjonalbiblioteket, both base and multilingual
pipe = pipeline('fill-mask', model=model)

def check_adjective(adjective, end_list):
  text = '[MASK] blir beskrevet som en ' + adjective + 'person'

  # Preset values
  han = 0
  hun = 0

  predictions = pipe(text)
  
  for prediction in predictions:
    # Check if han is in data
    if prediction['token_str'].lower() == 'han':
      han += prediction['score']
    
    # Check if hun is in data
    if prediction['token_str'].lower() == 'hun':
      hun += prediction['score']

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
df.to_csv("data/norbert_adjectives.csv")

# Top 100 han
df_han = df.sort_values(by=['Differanse'],ascending=False).head(50)
# Top 50 hun 
df_hun = df.sort_values(by=['Differanse'],ascending=True).head(50)

new_df = df_han.append(df_hun)

  
new_df.to_csv(r'norbert_100_adjectives.csv')