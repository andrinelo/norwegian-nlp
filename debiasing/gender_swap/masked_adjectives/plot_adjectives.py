from transformers import pipeline
from IPython.display import display

from transformers import logging
logging.set_verbosity_error()

import pandas as pd
from wordcloud import WordCloud


def get_data(model_name):
    df = pd.read_csv('debiasing/gender_swap/masked_adjectives/data/{}_100_adjectives.csv'.format(model_name))

    df_hun = pd.read_csv('debiasing/gender_swap/masked_adjectives/data/{}_female_adjectives.csv'.format(model_name))
    df_han = pd.read_csv('debiasing/gender_swap/masked_adjectives/data/{}_male_adjectives.csv'.format(model_name))

    return df_han, df_hun



def get_word_cloud_string(df):

    df_rounded= df.round(decimals=3)
    word_cloud_string = ''
    for index, row in df_rounded.iterrows(): 
        words = [row['Adjektiv'] for i in range(1)]
        string = ' '.join(words) + ' '
        word_cloud_string += string
    return word_cloud_string


def plot_word_cloud(word_cloud_string, model_name, male_or_female):
    if (male_or_female == 'male'):
        word_cloud = WordCloud(width = 250, height = 360, collocations=False, background_color='white', colormap='viridis').generate(word_cloud_string)
        word_cloud.to_file("debiasing/gender_swap/masked_adjectives/results/word_cloud_{}_{}.png".format(male_or_female, model_name))
        word_cloud.to_file("debiasing/gender_swap/masked_adjectives/results/word_cloud_{}_{}.eps".format(male_or_female, model_name))
    else:
        word_cloud = WordCloud(width = 250, height = 360, collocations=False, background_color='white', colormap='plasma').generate(word_cloud_string)
        word_cloud.to_file("debiasing/gender_swap/masked_adjectives/results/word_cloud_{}_{}.png".format(male_or_female, model_name))
        word_cloud.to_file("debiasing/gender_swap/masked_adjectives/results/word_cloud_{}_{}.eps".format(male_or_female, model_name))

def run(df, model_name, is_male):
    if is_male ==True:  
        string = get_word_cloud_string(df)
        #print(string)
        plot_word_cloud(string, model_name, 'male')
    else: 
        string = get_word_cloud_string(df)
        plot_word_cloud(string, model_name, 'female')



if __name__ == '__main__': 

    name = 'male2female'


    df_male, df_female = get_data(name)
    run(df_male, name, True)
    run(df_female, name, False)
