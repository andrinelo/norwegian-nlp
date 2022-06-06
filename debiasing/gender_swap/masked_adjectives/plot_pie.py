

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def plot_pie(results, labels, save_to):
    fig, ax = plt.subplots(figsize=(3.5, 3.7), subplot_kw=dict(aspect="equal"))
    colors = {'Female': "#AADEA7", 'Male': "#2D87BB", 'Excluded from results': "#E6F69D"}

    wedges, texts, autotexts = ax.pie(results, autopct='%1.1f%%', textprops=dict(
        color="black"), colors=[colors[v] for v in labels])

    ax.legend(wedges, labels,
              loc=1)

    plt.setp(autotexts, size=10)

    #ax.set_title(title, fontsize=10)

    plt.savefig(save_to)

male2female = np.array([0, 490, 1230])

labels = ["Male", "Female", "Excluded from results"]

#title = 'Number of adjectives biased in male and female \n direction as predicted by NB-BERT-male2female'


NB_BERTsave_eps = 'debiasing/gender_swap/masked_adjectives/results/plot_pie_NB_BERT_male2female.eps'
NB_BERTsave_png = 'debiasing/gender_swap/masked_adjectives/results/plot_pie_NB_BERT_male2female.png'


plot_pie(male2female, labels, NB_BERTsave_eps)
plot_pie(male2female, labels, NB_BERTsave_png)