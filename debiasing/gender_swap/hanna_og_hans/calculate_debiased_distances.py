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

def get_pca_emb(emb_feat_dF, n, model_name): 
    #calculate PCA of top n features
    pca_emb = PCA(n_components=n)
    principalComponents_emb = pca_emb.fit_transform(emb_feat_dF)
    principal_emb_Df = pd.DataFrame(data = principalComponents_emb, columns = ['pc {}'.format(i) for i in range(1, n+1)])
    print(principal_emb_Df.tail())

    print('Explained variation per principal component: {}'.format(pca_emb.explained_variance_ratio_))

    if n==10:
        return pca_emb
    
    return torch.from_numpy(pca_emb.components_)

def extract_sentences():
    df = pd.read_excel('data_sets/hanna_og_hans_vurderinger.xlsx', sheet_name='Ark 1')
    sentences = df['Setning'].values
    return sentences

def get_hans_hanna_emb(model_name, type_of_embedding): 
    hans = np.loadtxt('debiasing/gender_swap/hanna_og_hans/data/{}_{}_{}.txt'.format(model_name, 'hans', type_of_embedding))
    hanna = np.loadtxt('debiasing/gender_swap/hanna_og_hans/data/{}_{}_{}.txt'.format(model_name, 'hanna', type_of_embedding))
    return hans, hanna

def get_gender_subspace_emb(model_name): 

    diff_embeddings = np.loadtxt('debiasing/gender_swap/hanna_og_hans/data/diff_embeddings_{}.txt'.format(model_name), delimiter=' ')

    emb_feat_dF = get_feat_dF(diff_embeddings)
    pca_emb_2 = get_pca_emb(emb_feat_dF, 2, model_name)
    pca_emb_10 = get_pca_emb(emb_feat_dF, 10, model_name)
    plot_bar(pca_emb_10, model_name, 10)

    return pca_emb_2


def get_emb_test_sentences(NorBERT): 
    df = pd.read_excel('data_sets/hanna_og_hans_vurderinger.xlsx', sheet_name='Ark 1')
    sentences = df['Setning'].values
    sentence_embeddings = [extract_total_embedding_from_text(sentence, NorBERT) for sentence in sentences]
    return sentence_embeddings


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


def plot(diff_list_norbert, diff_list_nb_bert, diff_list_mbert, diff_list_male2female, sentences, type_of_embeddings, when):

    data = {'NorBERT': diff_list_norbert, 'NB-BERT': diff_list_nb_bert, 'mBERT': diff_list_mbert, 'male2female': diff_list_male2female}
    df = pd.DataFrame(data,columns=['NorBERT','NB-BERT', 'mBERT', 'male2female'], index = ['S{}'.format(i) for i in range(1, len(sentences)+1)])
    #df = pd.DataFrame(data,columns=['NorBERT','NB-BERT', 'mBERT'], index = sentences)

    plt.style.use('ggplot')

    df.plot.barh()

    plt.ylabel('Sentence used to calculate similarity',fontsize=10, labelpad=2)
    plt.xlabel('Cosine similarity difference (male-female)',fontsize=10)
    #plt.title("Results for [{}]\n".format(type_of_embeddings),fontsize=15)

    #plt.show()

    save_plot(plt, 'debiasing/gender_swap/hanna_og_hans/results/diff_plot'+when+type_of_embeddings+'.eps')
    save_plot(plt, 'debiasing/gender_swap/hanna_og_hans/results/diff_plot'+when+type_of_embeddings+'.png')



def plot_bar(pca_emb, model_name, number_of_features): 
    plt.figure()
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component',fontsize=20)
    plt.ylabel('Percentage of explained variation',fontsize=20)

    x = [('PC' + str(i)) for i in range(1, number_of_features+1)]
    y = [pca_emb.explained_variance_ratio_[i] for i in range(number_of_features)]
    plt.bar(x, y, color='steelblue')

    save_plot(plt, 'debiasing/gender_swap/hanna_og_hans/results/PCA_{}.png'.format(model_name))
    save_plot(plt, 'debiasing/gender_swap/hanna_og_hans/results/PCA_{}.eps'.format(model_name))


if __name__ == '__main__': 

    NorBERT = 'ltgoslo/norbert'
    NB_BERT = 'NbAiLab/nb-bert-base'
    mBERT = 'bert-base-multilingual-cased'
    male2female = 'NbAiLab/nb-bert-ncc-male2female'


    model_list = [NorBERT, NB_BERT, mBERT, male2female]
    model_name = ['NorBERT', 'NB-BERT', 'mBERT', 'male2female']

    sentences = extract_sentences()
    test_sentence_embeddings = get_emb_test_sentences(NorBERT)
    
    type_of_embeddings = ['SA', 'TWA']
   
    for type in type_of_embeddings: 
        diffs = []
        diffs_debiased = []
        for i in range(len(model_list)):
            
            pca_embedding = get_gender_subspace_emb(model_name[i])
            hans, hanna = get_hans_hanna_emb(model_name[i], type)

            diff_list = get_similarity(test_sentence_embeddings, hanna, hans, type, model_name[i], 'debiasing/gender_swap/hanna_og_hans/results/diffs.txt')
            
            diffs.append(diff_list)
        plot(diffs[0], diffs[1], diffs[2], diffs[3], sentences, type, 'original')
