from transformers import pipeline
from IPython.display import display

from transformers import logging
logging.set_verbosity_error()

import pandas as pd
#from wordcloud import WordCloud


def get_data(model_name):
    df = pd.read_csv('experiments/masked_adjectives/{}_100_adjectives.csv'.format(model_name))

    df_male = df.iloc[:50]
    df_female = df.iloc[50:]

    return df_male, df_female



def get_word_cloud_string(df):

    df_rounded= df.round(decimals=3)
    word_cloud_string = ''

    for index, row in df_rounded.iterrows(): 
        print(row['Adjektiv'], row['Differanse'])
        words = [row['Adjektiv'] for i in range(int(row['Differanse']*1000))]
        print(words)
        string = ' '.join(words) + ' '
        word_cloud_string += string
    return word_cloud_string


def plot_word_cloud(word_cloud_string, model_name, male_or_female): 
    word_cloud = WordCloud(collocations=False, background_color='white').generate(word_cloud_string)
    word_cloud.to_file("word_cloud_{}_{}.png".format(male_or_female, model_name))
    word_cloud.to_file("word_cloud_{}_{}.eps".format(male_or_female, model_name))

def run(df, model_name, is_male):
    if is_male ==True:  
        string = get_word_cloud_string(df)
        print(string)
        plot_word_cloud(string, model_name, 'male')
    else: 
        string = get_word_cloud_string(df)
        plot_word_cloud(string, model_name, 'female')



if __name__ == '__main__': 

    name_list = ['NorBERT', 'NB-BERT', 'mBERT']

    df_male, df_female = get_data('norbert')

    for model_name in name_list:

        run(df_male, model_name, True)
        run(df_female, model_name, False)
