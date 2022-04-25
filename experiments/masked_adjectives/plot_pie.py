

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def plot_pie(results, labels, title, save_to):
    fig, ax = plt.subplots(figsize=(3.5, 3.7), subplot_kw=dict(aspect="equal"))
    colors = {'Female': "#AADEA7", 'Male': "#2D87BB", 'Excluded from results': "#E6F69D"}

    wedges, texts, autotexts = ax.pie(results, autopct='%1.1f%%', textprops=dict(
        color="black"), colors=[colors[v] for v in labels])

    ax.legend(wedges, labels,
              loc=1)

    plt.setp(autotexts, size=10)

    ax.set_title(title, fontsize=11)

    plt.savefig(save_to)


NorBERT = np.array([1549, 66, 105])
NB_BERT = np.array([1263, 431, 26])
mBERT = np.array([100, 0, 24])

labels = ["Male", "Female", "Excluded from results"]

titleNorBERT = 'Number of adjectives biased in male and \n female direction as predicted by NorBERT'
titleNB_BERT = 'Number of adjectives biased in male and \n female direction as predicted by NB_BERT'
titlemBERT = 'Number of adjectives biased in male and \n female direction as predicted by mBERT'


NorBERTsave = 'experiments\masked_adjectives\plot_pie_NorBERT.eps'
NB_BERTsave = 'experiments\masked_adjectives\plot_pie_NB_BERT.eps'
mBERTsave = 'experiments\masked_adjectives\plot_pie_mBERT.eps'

plot_pie(NorBERT, labels, titleNorBERT, NorBERTsave)
plot_pie(NB_BERT, labels, titleNB_BERT, NB_BERTsave)
plot_pie(mBERT, labels, titlemBERT, mBERTsave)