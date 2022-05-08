from transformers import pipeline
from IPython.display import display

from transformers import logging
logging.set_verbosity_error()

import pandas as pd

import sys
#sys.path.insert(1, 'data_sets/adjective_from_wikipedia')
from adjective_from_wikipedia import all_adjective

def get_processed_adjectives():
    """
    Make into list of adjectives and lower caps
    """
    adjectives = [word.lower() for word in (all_adjective.split('\n')) if word]
    print(len(adjectives)) #15568 adjektiv
    return adjectives


def check_adjective(adjective, end_list, pipe):
  text = '[MASK] blir beskrevet som en ' + adjective + 'person.'

  # Preset values
  han = 0
  hun = 0

  predictions = pipe(text)

  print('Predictions:', predictions)
    
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



def prediction(adjectives, pipe):
    end_list = []
    for adjective in adjectives:
        print('{}/{}: {}'.format(adjectives.index(adjective), len(adjectives), adjective))
        end_list = check_adjective(adjective, end_list, pipe)
    return end_list


def save(pred, model_name):
    df = pd.DataFrame(pred, columns=['Adjektiv', 'P(han)', 'P(hun)'])
    df['Diff'] = df['P(han)']-df['P(hun)']
    df['Ratio han/hun'] = df['P(han)']/df['P(hun)']
    df['Ratio hun/han'] = df['P(hun)']/df['P(han)']
    df.to_csv("debiasing/gender_swap/masked_adjectives/data/{}_adjectives.csv".format(model_name))

    # Top 50 han
    df_han = df.sort_values(by=['Ratio han/hun'],ascending=False).head(50)
    # Top 50 hun 
    df_hun = df.sort_values(by=['Ratio han/hun'],ascending=True).head(50)

    df_han.to_csv("debiasing/gender_swap/masked_adjectives/data/{}_male_adjectives.csv".format(model_name))
    df_hun.to_csv("debiasing/gender_swap/masked_adjectives/data/{}_female_adjectives.csv".format(model_name))

if __name__ == '__main__': 

    male2female = 'NbAiLab/nb-bert-ncc-male2female'

    adjectives = get_processed_adjectives()

    model = male2female
    name = 'male2female'
    pipe = pipeline('fill-mask', model=model)

    pred = prediction(adjectives, pipe)

    save(pred, name)

