from transformers import pipeline
from process_professions import professions

import pandas as pd

model = 'ltgoslo/norbert' #can also check with models from Nasjonalbiblioteket, both base and multilingual
pipe = pipeline('fill-mask', model=model)

def check_profession(profession):
  text = '[MASK] er ' + profession

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
      end_list.append([profession, han, hun])
    return end_list

def prediction():
    end_list = []
    for profession in professions:
        end_list = check_profession(profession, end_list)
    return end_list

prediction()
#df = pd.DataFrame(prediction(), columns=['Yrke', 'P(han)', 'P(hun)'])
#df['Differanse'] = df['Han']-df['Hun']
#Display(df)

# Top 50 hun
#df.sort_values(by=['Diff'],ascending=False).head(50)
# Top 50 han 
#df.sort_values(by=['Diff'],ascending=False).head(50)
