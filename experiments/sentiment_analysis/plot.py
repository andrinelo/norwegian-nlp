

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
import sys
sys.path.insert(1, 'experiments/handle_embeddings')
from extract_embeddings import *
"""


def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


def plot_pie(results, labels, title, save_to, save_to2):
    fig, ax = plt.subplots(figsize=(3.5, 3.7), subplot_kw=dict(aspect="equal"))
    colors = {'Hanna': "#AADEA7", 'Hans': "#2D87BB", 'Neutral': "#E6F69D"}

    wedges, texts, autotexts = ax.pie(results, autopct='%1.1f%%', textprops=dict(
        color="black"), colors=[colors[v] for v in labels.value_counts().keys()])

    ax.legend(wedges, labels,
              loc=1)

    plt.setp(autotexts, size=10)

    ax.set_title(title)

    plt.savefig(save_to)
    plt.savefig(save_to2)
    # plt.savefig('experiments/sentiment_analysis/results_figs/sentiment_analysis_pos_norbert.eps')
    # plt.savefig('experiments/sentiment_analysis_neg.png')
    # plt.savefig('experiments/sentiment_analysis_neg.eps')


def get_and_clean_data(modelname):
    df = pd.read_csv(
        'experiments/sentiment_analysis/{}/hanna_hans_analyze.csv'.format(modelname))
    df.drop(['Hans', 'Hanna_Softmax', 'Hans_Softmax', 'Diff pos sentiment',
            'Diff neg sentiment', 'Pos polarity', 'Neg polarity'], axis=1, inplace=True)
    df.columns = ['SN', 'ID2', 'Hanna/Hans Sentence', 'Hanna Logit',
                  'Hans Logit', 'Hanna Sentiment', 'Hans Sentiment']  # ['SN' + str(i) for i in range(len(df))] instead of SN for sentence number ask Reggy
    df.drop(['ID2'], axis=1, inplace=True)
    return df


def check_sentiment(modelname):
    equal_counter = 0
    df = get_and_clean_data(modelname)
    df['Diff'] = np.where(df['Hanna Sentiment'] ==
                          df['Hans Sentiment'], '0', '1')
    new = df.loc[~(df['Hanna Sentiment'] == df['Hans Sentiment'])]
    new.drop([''])
    print(new[])


if __name__ == '__main__':
    name_list = ['mBERT', 'NB-BERT', 'NorBERT']

    for name in name_list:
        print(name)
        check_sentiment(name)

    """
    for name in name_list:
        df_positive=pd.read_csv(
            'experiments/sentiment_analysis/{}/who_is_more_positive.csv'.format(name))
        results = df_positive['Total sentences pos']
        labels = df_positive['Who is more positive']
        title = "Share of sentences that are perceived \n as more positive by {}".format(
            name)
        save_to = 'experiments/sentiment_analysis/results_fig/sentiment_analysis_pos_{}.png'.format(
            name)
        save_to2 = 'experiments/sentiment_analysis/results_fig/sentiment_analysis_pos_{}.eps'.format(
            name)
        plot_pie(results, labels, title, save_to, save_to2)

        df_negative = pd.read_csv(
            'experiments/sentiment_analysis/{}/who_is_more_negative.csv'.format(name))
        results = df_negative['Total sentences neg']
        labels = df_negative['Who is more negative']
        title = "Share of sentences that are perceived \n as more negative by {}".format(
            name)
        save_to = 'experiments/sentiment_analysis/results_fig/sentiment_analysis_neg_{}.png'.format(
            name)
        save_to2 = 'experiments/sentiment_analysis/results_fig/sentiment_analysis_neg_{}.eps'.format(
            name)
        plot_pie(results, labels, title, save_to, save_to2)
    """
