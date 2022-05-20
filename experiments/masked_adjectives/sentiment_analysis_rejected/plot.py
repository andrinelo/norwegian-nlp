

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


def get_and_clean_data(modelname):
    df = pd.read_csv(
        'experiments/sentiment_analysis/{}/hanna_hans_analyze.csv'.format(modelname))
    df.drop(['Hans', 'Hanna_Softmax', 'Hans_Softmax', 'Diff pos sentiment',
            'Diff neg sentiment', 'Pos polarity', 'Neg polarity'], axis=1, inplace=True)
    df.columns = ['SN', 'ID2', 'Hanna/Hans Sentence', 'Hanna Logit',
                  'Hans Logit', 'Hanna Sentiment', 'Hans Sentiment']  # ['SN{}'.format(i) for i in range(1, len(sentences)+1)]
    df.drop(['ID2'], axis=1, inplace=True)
    return df


def check_sentiment(modelname):
    equal_counter = 0
    df = get_and_clean_data(modelname)
    df['Diff'] = np.where(df['Hanna Sentiment'] ==
                          df['Hans Sentiment'], '0', '1')
    new = df.loc[~(df['Hanna Sentiment'] == df['Hans Sentiment'])]
    new.drop([''])


def get_data(name):
    df_hun = pd.read_csv(
        'experiments/sentiment_analysis/{}/adjective_analyzed_hun.csv'.format(name))
    df_han = pd.read_csv(
        'experiments/sentiment_analysis/{}/adjective_analyzed_han.csv'.format(name))
    return df_hun, df_han


def get_sentiment_values(name):
    df_values_hun = df_hun[['Complete sentence female', 'Sentiment']].copy()
    df_values_han = df_han[['Complete sentence male', 'Sentiment']].copy()
    hun_positive = df_values_hun.Sentiment.str.count("Positive").sum()
    hun_negative = df_values_hun.Sentiment.str.count("Negative").sum()
    han_positive = df_values_han.Sentiment.str.count("Positive").sum()
    han_negative = df_values_han.Sentiment.str.count("Negative").sum()

    print(name)
    print(hun_positive, hun_negative, hun_negative+hun_positive,
          "Should sum to 431 for NB-BERT and 66 for NorBERT")
    print(han_positive, han_negative, han_negative+han_positive,
          "Should sum to 1263 for NB-BERT and 1549 for NorBERT")

    # return hun_positive, hun_negative, han_positive, han_negative


def plot_bar():
    labels = ['Positive sentences', 'Negative sentences']
    women = [430, 1]
    men = [1258, 5]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    colours = {'Female': "#AADEA7", 'Male': "#2D87BB"}

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, men, width, label='Male', color="#2D87BB")
    rects2 = ax.bar(x + width/2, women, width,
                    label='Female', color='#AADEA7')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('NB-BERT')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.savefig(
        "experiments/sentiment_analysis/results_fig/count_adjective_NB-BERT.eps")
    plt.savefig(
        "experiments/sentiment_analysis/results_fig/count_adjective_NB-BERT.png")


def plot_bar_norbert():
    labels = ['Positive sentences', 'Negative sentences']
    women = [12, 54]
    men = [253, 1296]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, men, width, label='Male', color="#2D87BB")
    rects2 = ax.bar(x + width/2, women, width, label='Female', color='#AADEA7')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('NorBERT')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.savefig(
        "experiments/sentiment_analysis/results_fig/count_adjective_NorBERT.eps")
    plt.savefig(
        "experiments/sentiment_analysis/results_fig/count_adjective_NorBERT.png")


if __name__ == '__main__':
    plot_bar()
    plot_bar_norbert()
    """
    name_list = ['NB-BERT', 'NorBERT']  # ['mBERT', 'NB-BERT', 'NorBERT']
    for name in name_list:
        df_hun, df_han = get_data(name)
        get_sentiment_values(name)


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
