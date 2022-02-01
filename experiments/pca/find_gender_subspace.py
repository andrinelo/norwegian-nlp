from tkinter import N
from transformers import pipeline

from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
Following code retrieved (copied) from https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d
"""

def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors

def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings

# Text corpus
##############
# These sentences show the different
# forms of the word 'han/hun' to show the
# value of contextualized embeddings

def get_embeddings_from_text(texts, gender_word, model, tokenizer):


    # Getting embeddings for the target
    # word in all given contexts
    target_word_embeddings = []

    for text in texts:
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)

        #print('Tokenized text: ', tokenized_text)
        
        # Find the position 'bank' in list of tokens
        word_index = tokenized_text.index(gender_word) # = dét ordet som er i alle setningne i form av ulike kontekster
        # Get the embedding for bank
        word_embedding = list_token_embeddings[word_index]


        target_word_embeddings.append(word_embedding) #= embeddings for "bank" i kontekst av de ulike setningene
    #print(target_word_embeddings)
    return target_word_embeddings


#get average vector of "han". switch row and cols to sum each colums and take average
"""
def get_average(target_word_embeddings, han_hun): 
    switch_rows_and_cols_vector = [[] for i in range(len(target_word_embeddings[0]))]

    for number_index in range(len(target_word_embeddings[0])):
        for vector_index in range(len(target_word_embeddings)):
            switch_rows_and_cols_vector[number_index].append(target_word_embeddings[vector_index][number_index])
    print('Length of new vector: ', len(switch_rows_and_cols_vector))

    avg_vector = []
    for row in switch_rows_and_cols_vector: 
        npArray = np.array(row)
        avg = np.average(npArray)
        avg_vector.append(avg)
    print('Average vector for ', han_hun, ':', avg_vector)
    return avg_vector
"""

def make_numpy(vector): 
    return np.array(vector)

def find_difference(han_vector, hun_vector):
    han = make_numpy(han_vector)
    hun = make_numpy(hun_vector)
    diffs = hun-han
    return diffs

def get_pca_emb(emb_feat_dF, n): 
    #calculate PCA of top 10 features
    pca_emb = PCA(n_components=n)
    principalComponents_emb = pca_emb.fit_transform(emb_feat_dF)
    principal_emb_Df = pd.DataFrame(data = principalComponents_emb, columns = ['pc {}'.format(i) for i in range(1, n+1)])
    print(principal_emb_Df.tail())

    print('Explained variation per principal component: {}'.format(pca_emb.explained_variance_ratio_))

    return pca_emb


def get_feat_dF(embeddings): 
    # normalizing the features
    embeddings_standarized = StandardScaler().fit_transform(embeddings) 
    print('Mean and std.div: ', np.mean(embeddings_standarized), np.std(embeddings_standarized))
    
    #create data frame with features
    emb_feat_cols = ['feature'+str(i) for i in range(embeddings_standarized.shape[1])]
    emb_feat_dF = pd.DataFrame(embeddings_standarized, columns=emb_feat_cols)
    print(emb_feat_dF)

    return emb_feat_dF

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

def plot_bar(pca_emb): 
    plt.figure()
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component',fontsize=20)
    plt.ylabel('Percentage of explained variation',fontsize=20)
    plt.title("Explained variation per principal component",fontsize=20)

    x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    y = [pca_emb.explained_variance_ratio_[0], pca_emb.explained_variance_ratio_[1], pca_emb.explained_variance_ratio_[2], pca_emb.explained_variance_ratio_[3],
    pca_emb.explained_variance_ratio_[4], pca_emb.explained_variance_ratio_[5], pca_emb.explained_variance_ratio_[6], pca_emb.explained_variance_ratio_[7], 
    pca_emb.explained_variance_ratio_[8], pca_emb.explained_variance_ratio_[9]] 
    plt.bar(x, y)
    plt.show()

def extract_sentences(filename, sheetname):

    df = pd.read_excel(filename, sheet_name=sheetname)

    han_sentences = df['Han'].values
    hun_sentences = df['Hun'].values

    return han_sentences, hun_sentences

def run(sentence_path, sheet_name, gender_word_pair_male, gender_word_pair_female, model_name, number_of_features):
    han, hun = extract_sentences(filename=sentence_path, sheetname=sheet_name)

    texts_han = [gender_word_pair_male] + han
    texts_hun = [gender_word_pair_female] + hun

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
    tokenizer = BertTokenizer.from_pretrained(model_name)


    han_embeddings = make_numpy(get_embeddings_from_text(texts_han, gender_word_pair_male, model, tokenizer)).transpose()
    hun_embeddings = make_numpy(get_embeddings_from_text(texts_hun, gender_word_pair_female, model, tokenizer)).transpose()

    diff_embeddings = han_embeddings-hun_embeddings
    #print('Diff embedding: ', diff_embeddings)

    emb_feat_dF = get_feat_dF(diff_embeddings)

    pca_emb = get_pca_emb(emb_feat_dF, number_of_features)

    #plot results
    plot_bar(pca_emb)
    #plot_scatter(principal_emb_Df)



if __name__ == '__main__':

    NorBERT = 'ltgoslo/norbert'
    NB_BERT = 'NbAiLab/nb-bert-base'
    mBERT = 'bert-base-multilingual-cased'

    #fill in the following variables to plot pca of top n features in sheet in .xlsx file calculated with one of the norwegian language models

    sentence_path = 'experiments\pca\sample_sentences.xlsx'
    sheet_name = 'hun_han_alle'
    gender_word_pair_male = 'han'
    gender_word_pair_female = 'hun'

    model_name = mBERT
    number_of_features = 10

    run(
        sentence_path=sentence_path, 
        sheet_name=sheet_name, 
        gender_word_pair_male=gender_word_pair_male, 
        gender_word_pair_female=gender_word_pair_female, 
        model_name=model_name, 
        number_of_features=number_of_features)

    
