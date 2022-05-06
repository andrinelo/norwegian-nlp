from matplotlib.pyplot import axis
import pandas as pd
import json

import sys
sys.path.insert(1, 'data_sets/adjective_from_wikipedia')
from adjective_from_wikipedia import all_adjective


def get_number_of_adjectives():
    """
    Make into list of adjectives and lower caps
    """
    adjectives = [word.lower() for word in (all_adjective.split('\n')) if word]
    return len(adjectives) 

def get_data(model_name):
    df = pd.read_csv('experiments/masked_adjectives/data/{}_adjectives.csv'.format(model_name))
    df_hun = df[(df['Diff'] < 0)].sort_values(by=['Diff'],ascending=True)
    df_han = df[(df['Diff'] > 0)].sort_values(by=['Diff'],ascending=False)
    return df_han, df_hun

def get_top_n_values(df, who, n=None):
    if who == 'han':
        if n is not None:
            df = df[:n]
        values_list = [item for item in df['Ratio han/hun']]
        summarize = sum(values_list)/df.shape[0]
        return summarize
    if who == 'hun':
        if n is not None:
            df = df[:n]
        values_list = [item for item in df['Ratio hun/han']]
        summarize = sum(values_list)/df.shape[0]
        return summarize

def calculate_scores(df, who): 

    if df.shape[0] == 0: 
        return {'Number of adjectives came out as biased': '0/{}'.format(get_number_of_adjectives())}

    scores_1 = get_top_n_values(df, who, 1)
    scores_3 = get_top_n_values(df, who, 3)
    scores_50 = get_top_n_values(df, who, 50)
    scores_agg = get_top_n_values(df, who)
    dict = {'Number of adjectives came out as biased': '{}/{}'.format(df.shape[0], get_number_of_adjectives()),'Top 1': scores_1, 'Top 3 average': scores_3, 'Top 50 average': scores_50, 'Aggregated average': scores_agg} 
    return dict


if __name__ == '__main__': 

    name_list = ['NorBERT', 'NB-BERT', 'mBERT']

    for model_name in name_list:
        df_han, df_hun = get_data(model_name)
        scores_han = calculate_scores(df_han, 'han')
        scores_hun = calculate_scores(df_hun, 'hun')
        """
        print('\n', model_name)
        print('Male')
        for key, value in scores_han.items(): 
            print(key, value)
        print('Female')
        for key, value in scores_hun.items(): 
            print(key, value)
        """
        with open("experiments/masked_adjectives/results/scores_{}.txt".format(model_name), 'w') as file:
            file.write('Male: ' + json.dumps(scores_han) + '\nFemale: ' + json.dumps(scores_hun))