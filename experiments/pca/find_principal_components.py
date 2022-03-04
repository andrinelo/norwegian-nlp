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

#TODO import .handle_embeddings.extract_embeddings here



def get_feat_dF(embeddings): 
    # normalizing the features
    embeddings_standarized = StandardScaler().fit_transform(embeddings) 
    print('Mean and std.div: ', np.mean(embeddings_standarized), np.std(embeddings_standarized))
    
    #create data frame with features
    emb_feat_cols = ['feature'+str(i) for i in range(embeddings_standarized.shape[1])]
    emb_feat_dF = pd.DataFrame(embeddings_standarized, columns=emb_feat_cols)
    #print(emb_feat_dF)

    return emb_feat_dF

def get_pca_emb(emb_feat_dF, n): 
    #calculate PCA of top 10 features
    pca_emb = PCA(n_components=n)
    principalComponents_emb = pca_emb.fit_transform(emb_feat_dF)
    principal_emb_Df = pd.DataFrame(data = principalComponents_emb, columns = ['pc {}'.format(i) for i in range(1, n+1)])
    print(principal_emb_Df.tail())

    print('Explained variation per principal component: {}'.format(pca_emb.explained_variance_ratio_))

    return pca_emb

def plot_scatter(principal_emb_Df): 
    plt.figure()
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1',fontsize=20)
    plt.ylabel('Principal Component - 2',fontsize=20)
    plt.title("Principal Component Analysis of Norwegian BERT word embeddings",fontsize=20)
    plt.scatter(principal_emb_Df['principal component 1'], principal_emb_Df['principal component 2'], c = np.random.rand(768), s = 10)
    plt.show()  

def plot_bar(pca_emb, model_name): 
    plt.figure()
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component',fontsize=20)
    plt.ylabel('Percentage of explained variation',fontsize=20)
    plt.title("Explained variation per principal component in \n {}".format(str(model_name)),fontsize=20)

    x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    y = [pca_emb.explained_variance_ratio_[0], pca_emb.explained_variance_ratio_[1], pca_emb.explained_variance_ratio_[2], pca_emb.explained_variance_ratio_[3],
    pca_emb.explained_variance_ratio_[4], pca_emb.explained_variance_ratio_[5], pca_emb.explained_variance_ratio_[6], pca_emb.explained_variance_ratio_[7], 
    pca_emb.explained_variance_ratio_[8], pca_emb.explained_variance_ratio_[9]] 
    plt.bar(x, y)
    #plt.show()

    return plt

def extract_sentences(filename, sheetname):

    df = pd.read_excel(filename, sheet_name=sheetname)

    han_sentences = df['Han'].values
    hun_sentences = df['Hun'].values

    return han_sentences, hun_sentences

def save_plot(plot, plot_to_filename): 
    plot.savefig(plot_to_filename)


def run(sentence_path, sheet_name, gender_word_pair_male, gender_word_pair_female, model_name, number_of_features, plot_to_filename):
   
    han_sentences, hun_sentences = extract_sentences(filename=sentence_path, sheetname=sheet_name)

    # Loading the pre-trained BERT model
    ###################################
    # Embeddings will be derived from
    # the outputs of this model
    model = BertModel.from_pretrained(model_name, output_hidden_states = True,)

    # Setting up the tokenizer
    ###################################
    # This is the same tokenizer that
    # was used in the model to generate
    # embeddings to ensure consistency
    tokenizer = BertTokenizer.from_pretrained('ltgoslo/norbert')


    #TODO Er veldig usikker på om output her skal være .transposed() eller ikke, altså hva som er rader og hva som er kolonner i embeddingsene
    # Se på data frame outputten for å se om den skal transposes...? 

    #TODO I tillegg så får jeg ikke til å slå sammen for hun/han og jente/gutt pga strl. på matrisen (antall setninger). 
    # Fikk ikke fikset det problemet uten å fikse problemet over, så må se senere om det er noe å få gjort med det.

    hun_embeddings = extract_average_embedding_for_specific_word_in_multiple_sentences(hun_sentences, model, gender_word_pair_female)
    han_embeddings = extract_average_embedding_for_specific_word_in_multiple_sentences(han_sentences, model, gender_word_pair_male)

    diff_embeddings = han_embeddings - hun_embeddings
    print('Diff embedding: ', diff_embeddings)
    emb_feat_dF = get_feat_dF(diff_embeddings)
    print(emb_feat_dF)


    pca_emb = get_pca_emb(emb_feat_dF, number_of_features)

    #plot results
    plot = plot_bar(pca_emb, model_name)
    
    #save plot as .png
    save_plot(plot, plot_to_filename)



if __name__ == '__main__':
    
    #fill in the following variables to plot pca of 
    # top n features in sheet in .xlsx file calculated
    # with one of the norwegian language models
    
    NorBERT = 'ltgoslo/norbert'
    NB_BERT = 'NbAiLab/nb-bert-base'
    mBERT = 'bert-base-multilingual-cased'

    models_list = [NorBERT, NB_BERT, mBERT]
    name = ['NorBERT', 'NB-BERT', 'mBERT']
    sentence_path = 'experiments\pca\sample_sentences.xlsx'
    number_of_features = 10
    
    
    sheet_name_list = ['hun_han_alle', 'jente_gutt_tilfeldig']
    
    for sheet in sheet_name_list: 
        sheet_name = sheet
        if sheet_name == 'hun_han_alle':
            gender_word_pair_male='han'
            gender_word_pair_female='hun'
        if sheet_name == 'jente_gutt_tilfeldig':
            gender_word_pair_male='gutt'
            gender_word_pair_female='jente'
        for model in models_list:
            model_name = model
            
            plot_to_filename = 'experiments\pca\plots\{}_{}.png'.format(name[models_list.index(model_name)], sheet_name)

            print('Running {} sheet with {} and target words {} and {}'.format(sheet_name, model_name, gender_word_pair_male, gender_word_pair_female))
            run(
                sentence_path=sentence_path, 
                sheet_name=sheet_name, 
                gender_word_pair_male=gender_word_pair_male, 
                gender_word_pair_female=gender_word_pair_female, 
                model_name=model_name, 
                number_of_features=number_of_features, 
                plot_to_filename=plot_to_filename)
    """
    sheet_name = 'jente_gutt_tilfeldig'
    gender_word_pair_male='gutt'
    gender_word_pair_female='jente'

    for model in models_list:
            model_name = model
            
            plot_to_filename = 'experiments\pca\plots\{}_{}_{}.png'.format(sheet_name, models_list.index(model_name), str(number_of_features))

            print('Running {} sheet with {} and target words {} and {}'.format(sheet_name, model_name, gender_word_pair_male, gender_word_pair_female))
            run(
                sentence_path=sentence_path, 
                sheet_name=sheet_name, 
                gender_word_pair_male=gender_word_pair_male, 
                gender_word_pair_female=gender_word_pair_female, 
                model_name=model_name, 
                number_of_features=number_of_features, 
                plot_to_filename=plot_to_filename)
    """     