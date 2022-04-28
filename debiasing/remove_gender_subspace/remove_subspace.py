from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import sys
sys.path.insert(1, 'experiments/handle_embeddings')
from extract_embeddings import *

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
    #calculate PCA of top n features
    pca_emb = PCA(n_components=n)
    principalComponents_emb = pca_emb.fit_transform(emb_feat_dF)
    principal_emb_Df = pd.DataFrame(data = principalComponents_emb, columns = ['pc {}'.format(i) for i in range(1, n+1)])
    print(principal_emb_Df.tail())

    print('Explained variation per principal component: {}'.format(pca_emb.explained_variance_ratio_))

    return pca_emb

def extract_sentences():
    df = pd.read_excel('debiasing/remove_gender_subspace/hanna_og_hans_vurderinger.xlsx', sheet_name='Ark 1')
    sentences = df['Setning'].values
    return sentences

def get_hans_hanna_emb(model_name, type_of_embedding): 
    hans = np.loadtxt('debiasing/remove_gender_subspace/{}_{}_{}.txt'.format(model_name, 'hans', type_of_embedding))
    hanna = np.loadtxt('debiasing/remove_gender_subspace/{}_{}_{}.txt'.format(model_name, 'hanna', type_of_embedding))
    return hans, hanna

def get_gender_subspace_emb(model_name, number_of_features): 

    diff_embeddings = np.loadtxt('debiasing/remove_gender_subspace/diff_embeddings_{}.txt'.format(model_name), delimiter=' ')

    print(len(diff_embeddings))

    emb_feat_dF = get_feat_dF(diff_embeddings)
    pca_emb = get_pca_emb(emb_feat_dF, number_of_features)

    plot = plot_bar(pca_emb, model_name, number_of_features)

    return pca_emb


def get_emb_test_sentences(NorBERT): 
    df = pd.read_excel('debiasing/remove_gender_subspace/hanna_og_hans_vurderinger.xlsx', sheet_name='Ark 1')
    sentences = df['Setning'].values
    sentence_embeddings = [extract_total_embedding_from_text(sentence, NorBERT) for sentence in sentences]
    return sentence_embeddings


def remove_gender_subspace(test_sentence_emb, gender_subspace_emb): 
    neutral_test_sentence_embeddings = [torch.sub(sentence_emb, gender_subspace_emb) for sentence_emb in test_sentence_emb]
    return neutral_test_sentence_embeddings


def calculate_cosine(emb1, emb2): 
    sim = cosine_similarity(emb1, emb2)
    return sim


def get_similarity(sentence_embeddings, embedding_female, embedding_male, type_of_embeddings, model_name, filename): 

    diff_list = []

    with open(filename, 'a') as file:
        file.write('\n ===== Results for model [{}] with [{}] ===== \n'.format(model_name, type_of_embeddings))
        for index in range(len(sentence_embeddings)): 
            female = calculate_cosine(sentence_embeddings[index], embedding_female)
            male = calculate_cosine(sentence_embeddings[index], embedding_male)
            diff_list.append(male-female)
            file.write('S{}'.format(index))
            file.write('Diff (M-F): {} Diff ratio (M/F): {},\n'.format(male-female, male/female))
        
    return diff_list


def save_plot(plot, plot_to_filename): 
    plot.savefig(plot_to_filename)


def plot(diff_list_norbert, diff_list_nb_bert, diff_list_mbert, sentences, type_of_embeddings):

    data = {'NorBERT': diff_list_norbert, 'NB-BERT': diff_list_nb_bert, 'mBERT': diff_list_mbert}
    df = pd.DataFrame(data,columns=['NorBERT','NB-BERT', 'mBERT'], index = ['S{}'.format(i) for i in range(1, len(sentences)+1)])
    #df = pd.DataFrame(data,columns=['NorBERT','NB-BERT', 'mBERT'], index = sentences)

    plt.style.use('ggplot')

    df.plot.barh()

    plt.ylabel('Sentence used to calculate similarity',fontsize=10, labelpad=2)
    plt.xlabel('Cosine similarity difference (male-female)',fontsize=10)
    plt.title("Results for [{}]\n".format(type_of_embeddings),fontsize=15)

    #plt.show()

    save_plot(plt, 'debiasing/remove_gender_subspace/diff_plot_'+type_of_embeddings+'.eps')
    save_plot(plt, 'debiasing/remove_gender_subspace/diff_plot_'+type_of_embeddings+'.png')





def plot_bar(pca_emb, model_name, number_of_features): 
    plt.figure()
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component',fontsize=20)
    plt.ylabel('Percentage of explained variation',fontsize=20)
    plt.title("Explained variation per principal component in \n {}".format(str(model_name)),fontsize=20)

    #x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] 
    x = [('PC' + str(i)) for i in range(1, number_of_features+1)]
    y = [pca_emb.explained_variance_ratio_[i] for i in range(number_of_features)]
    #y = [pca_emb.explained_variance_ratio_[0], pca_emb.explained_variance_ratio_[1], pca_emb.explained_variance_ratio_[2], pca_emb.explained_variance_ratio_[3],
    #pca_emb.explained_variance_ratio_[4], pca_emb.explained_variance_ratio_[5], pca_emb.explained_variance_ratio_[6], pca_emb.explained_variance_ratio_[7], 
    #pca_emb.explained_variance_ratio_[8], pca_emb.explained_variance_ratio_[9]] 
    plt.bar(x, y)

    save_plot(plt, 'debiasing/remove_gender_subspace/PCA_{}.png'.format(model_name))
    save_plot(plt, 'debiasing/remove_gender_subspace/PCA_{}.eps'.format(model_name))


if __name__ == '__main__': 

    NorBERT = 'ltgoslo/norbert'
    NB_BERT = 'NbAiLab/nb-bert-base'
    mBERT = 'bert-base-multilingual-cased'

    model_list = [NorBERT, NB_BERT, mBERT]
    model_name = ['NorBERT', 'NB-BERT', 'mBERT']
    number_of_features = [2, 1, 2]

    sentences = extract_sentences()
    diffs = []

    type_of_embeddings = ['SA', 'TWA']

    for type in type_of_embeddings: 
        for i in range(len(model_list)):

            test_sentence_embeddings = get_emb_test_sentences(NorBERT)
            gender_subspace = get_gender_subspace_emb(model_name[i], number_of_features[i])
            neutral_test_sentence_embeddings = remove_gender_subspace(test_sentence_embeddings, gender_subspace)
            

            hans, hanna = get_hans_hanna_emb(model_name[i], type)

            diff_list = get_similarity(neutral_test_sentence_embeddings, hanna, hans, type, model_name[i], 'debiasing/remove_gender_subspace/diffs.txt')

            diffs.append(diff_list)
    
        plot(diffs[0], diffs[1], diffs[2], sentences, type)