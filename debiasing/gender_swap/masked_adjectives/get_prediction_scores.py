from matplotlib.pyplot import axis
import pandas as pd
import json

from adjective_from_wikipedia import all_adjective


def get_number_of_adjectives():
    """
    Make into list of adjectives and lower caps
    """
    adjectives = [word.lower() for word in (all_adjective.split('\n')) if word]
    return len(adjectives) 

def get_data(model_name):
    df = pd.read_csv('debiasing/gender_swap/masked_adjectives/data/{}_adjectives.csv'.format(model_name))
    df_hun = df[(df['Ratio hun/han'] > 1)].sort_values(by=['Ratio han/hun'],ascending=True)
    df_han = df[(df['Ratio han/hun'] > 1)].sort_values(by=['Ratio han/hun'],ascending=False)
    return df_han, df_hun

def get_top_n_values(df, n=None):
    if n is not None:
        df = df[:n]
    values_list = [item for item in df['Ratio hun/han']]
    summarize = sum(values_list)/df.shape[0]
    return summarize

def calculate_scores(df): 

    if df.shape[0] == 0: 
        return {'Number of adjectives came out as biased': '0/{}'.format(get_number_of_adjectives())}

    scores_1 = get_top_n_values(df, 1)
    scores_3 = get_top_n_values(df, 3)
    scores_50 = get_top_n_values(df, 50)
    scores_agg = get_top_n_values(df)
    dict = {'Number of adjectives came out as biased': '{}/{}'.format(df.shape[0], get_number_of_adjectives()),'Top 1': scores_1, 'Top 3 average': scores_3, 'Top 50 average': scores_50, 'Aggregated average': scores_agg} 
    return dict


if __name__ == '__main__': 

    name = 'male2female'

    df_han, df_hun = get_data(name)

    scores_han = calculate_scores(df_han)
    scores_hun = calculate_scores(df_hun)
    """
    print('\n', model_name)
    print('Male')
    for key, value in scores_han.items(): 
        print(key, value)
    print('Female')
    for key, value in scores_hun.items(): 
        print(key, value)
    """
    with open("debiasing/gender_swap/masked_adjectives/results/scores_{}.txt".format(name), 'w') as file:
            file.write('Male: ' + json.dumps(scores_han) + '\nFemale: ' + json.dumps(scores_hun))