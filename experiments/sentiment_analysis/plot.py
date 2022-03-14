import sys
sys.path.insert(1, 'experiments/handle_embeddings')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from extract_embeddings import *



df_positive = pd.read_csv(
    'experiments/sentiment_analysis/who_is_more_positive_norbert.csv')
df_negative = pd.read_csv(
    'experiments/sentiment_analysis/who_is_more_negative_norbert.csv')


def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)

def plot2(results, labels, title_labels, title, save_to):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    colors = ["#AADEA7", "#2D87BB", "#E6F69D"]
    
    wedges, texts, autotexts = ax.pie(df_positive['Total sentences pos'], colors=colors, autopct='%1.1f%%',
                                  textprops=dict(color="black"))

    ax.legend(wedges, df_positive['Who is more positive'], title="Who is more positive",loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.setp(autotexts, size=10)

    ax.set_title("Sentences that are perceived as more positive by NorBERT")


    plt.savefig('experiments/sentiment_analysis/results_figs/sentiment_analysis_pos_norbert.png')
    plt.savefig('experiments/sentiment_analysis/results_figs/sentiment_analysis_pos_norbert.eps')
    #plt.savefig('experiments/sentiment_analysis_neg.png')
    #plt.savefig('experiments/sentiment_analysis_neg.eps')

def plot(results, labels, title_labels, title, save_to):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    colors = ["#AADEA7", "#2D87BB", "#E6F69D"]
    
    wedges, texts, autotexts = ax.pie(results, colors=colors, autopct='%1.1f%%',
                                  textprops=dict(color="black"))

    ax.legend(wedges, labels, title_labels,loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.setp(autotexts, size=10)

    ax.set_title(title)


    plt.savefig(save_to)
    #plt.savefig('experiments/sentiment_analysis/results_figs/sentiment_analysis_pos_norbert.eps')
    #plt.savefig('experiments/sentiment_analysis_neg.png')
    #plt.savefig('experiments/sentiment_analysis_neg.eps')


if __name__ == '__main__':
    results = df_positive['Total sentences pos']
    labels = df_positive['Who is more positive']
    plot(results, labels, "Who is more positive", "Sentences that are perceived as more positive by NorBERT", 'experiments/sentiment_analysis/results_figs/sentiment_analysis_pos_norbert.png')
