from tkinter import N
from transformers import pipeline

from transformers import BertTokenizer, BertModel
from transformers import logging
logging.set_verbosity_error()

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# from ..handle_embeddings import extract_embeddings

import sys
sys.path.insert(1, 'experiments/handle_embeddings')
from extract_embeddings import *



def extract_sentences(filename, sheetname):

    df = pd.read_excel(filename, sheet_name=sheetname)

    han_sentences = df['Han'].values
    hun_sentences = df['Hun'].values

    return han_sentences, hun_sentences


def get_diff_embeddings(sentence_path, sheet_name, gender_word_pair_male, gender_word_pair_female, model_name): 
    han_sentences, hun_sentences = extract_sentences(filename=sentence_path, sheetname=sheet_name)

    han_sentences = han_sentences
    hun_sentences = hun_sentences

    #TODO Er veldig usikker på om output her skal være .transposed() eller ikke, altså hva som er rader og hva som er kolonner i embeddingsene
    # Se på data frame outputten for å se om den skal transposes...? 

    hun_embeddings = extract_all_embeddings_for_specific_word_in_multiple_sentences(hun_sentences, model_name, gender_word_pair_female)
    han_embeddings = extract_all_embeddings_for_specific_word_in_multiple_sentences(han_sentences, model_name, gender_word_pair_male)

    diff_embeddings = han_embeddings - hun_embeddings 
    print('Diff embedding: ', diff_embeddings)
    return diff_embeddings


def run(sentence_path, models_list, number_of_features): 

    sheet_name_list = ['hun_han_alle', 'jente_gutt_tilfeldig']
    
    for model_name in models_list: 

        for sheet in sheet_name_list: 
            sheet_name = sheet
            
            if sheet_name == 'hun_han_alle':
                gender_word_pair_male='han'
                gender_word_pair_female='hun'
                print('Running {} sheet with {} and target words {} and {}'.format(sheet_name, model_name, gender_word_pair_male, gender_word_pair_female))
                diff_hun_han = get_diff_embeddings(sentence_path, sheet_name, gender_word_pair_male, gender_word_pair_female, model_name)
                to_filename = 'debiasing/remove_gender_subspace/data/embeddings_{}_{}.txt'.format(sheet_name, name[models_list.index(model_name)])
                np.savetxt(to_filename, diff_hun_han.numpy())
            
            if sheet_name == 'jente_gutt_tilfeldig':
                gender_word_pair_male='gutt'
                gender_word_pair_female='jente'
                print('Running {} sheet with {} and target words {} and {}'.format(sheet_name, model_name, gender_word_pair_male, gender_word_pair_female))
                diff_jente_gutt = get_diff_embeddings(sentence_path, sheet_name, gender_word_pair_male, gender_word_pair_female, model_name)
                to_filename = 'debiasing/remove_gender_subspace/data/embeddings_{}_{}.txt'.format(sheet_name, name[models_list.index(model_name)])
                np.savetxt(to_filename, diff_jente_gutt.numpy())

        diff_embeddings = torch.cat((diff_jente_gutt, diff_hun_han), dim=0)
        to_filename = 'debiasing/remove_gender_subspace/data/diff_embeddings_{}.txt'.format(name[models_list.index(model_name)])
        np.savetxt(to_filename, diff_embeddings.numpy())
                
            
        #plot_to_filename = 'experiments/pca/plots\{}.eps'.format(name[models_list.index(model_name)])
        #get_pca_combined(diff_embeddings, plot_to_filename, model_name, number_of_features)


if __name__ == '__main__':
    
    #fill in the following variables to plot pca of 
    # top n features in sheet in .xlsx file calculated
    # with one of the norwegian language models
    
    NorBERT = 'ltgoslo/norbert'
    NB_BERT = 'NbAiLab/nb-bert-base'
    mBERT = 'bert-base-multilingual-cased'

    models_list = [mBERT]
    name = ['mBERT']
    sentence_path = 'data_sets/sample_sentences.xlsx'
    number_of_features = 10 #antall komponenteer
    
    run(sentence_path, models_list, number_of_features)