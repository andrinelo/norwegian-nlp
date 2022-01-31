from tkinter import N
from transformers import pipeline

from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


model_NorBERT = 'ltgoslo/norbert'
model_NB_BERT = 'NbAiLab/nb-bert-base'
model_mBERT = 'bert-base-multilingual-cased'

#return masked modelling pipeline for imput model
def fill_mask_pipeline(model): 
    return pipeline('fill-mask', model=model)

#return 
def sentiment_analysis_pipeline(model): 
    return pipeline('sentiment-analysis', model=model)

#PCA.fit_transform([word_vectors_from_dataset])

"""
Following code retrieved (copied) from https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d
"""

# Loading the pre-trained BERT model
###################################
# Embeddings will be derived from
# the outputs of this model
model = BertModel.from_pretrained('ltgoslo/norbert', output_hidden_states = True,)
# Setting up the tokenizer
###################################
# This is the same tokenizer that
# was used in the model to generate
# embeddings to ensure consistency
tokenizer = BertTokenizer.from_pretrained('ltgoslo/norbert')

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
# forms of the word 'bank' to show the
# value of contextualized embeddings

def get_embeddings_from_text(texts, han_hun):

    # Getting embeddings for the target
    # word in all given contexts
    target_word_embeddings = []

    for text in texts:
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)

        #print('Tokenized text: ', tokenized_text)
        
        # Find the position 'bank' in list of tokens
        word_index = tokenized_text.index(han_hun) # = dét ordet som er i alle setningne i form av ulike kontekster
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

def calculate_pca(diff_vector): 
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(diff_vector)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    
    return principalDf

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

    """
    fig = plt.figure()
    fig.suptitle('Bar chart BERT')
    ax = fig.add_axes([0,0,1,1])
    x = ['principal component 1', 'principal component 2']
    y = [pca_emb.explained_variance_ratio_[0], pca_emb.explained_variance_ratio_[1]]
    ax.bar(x, y)
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    plt.show()
    """

if __name__ == '__main__':

    texts_han = ["han",
            "han er snekker.",
            "han er sykepleier.", 
            "i går var han der.",
            "han er kokk", 
            "er han ikke kul", 
            "jeg kjenner han", 
            "han elsker meg", 
            "jeg liker han", 
            "se på han"] # = ord å sammenlikne target word med
    
    texts_hun = ["hun",
            "hun er snekker.",
            "hun er sykepleier.", 
            "I går var hun der.",
            "hun er kokk", 
            "er hun ikke kul", 
            "jeg kjenner hun", 
            "hun elsker meg", 
            "jeg liker hun", 
            "se på hun"]

    han_embeddings = make_numpy(get_embeddings_from_text(texts_han, 'han'))
    hun_embeddings = make_numpy(get_embeddings_from_text(texts_hun, 'hun'))

    han_transposed = han_embeddings.transpose()
    hun_transposed = hun_embeddings.transpose()

    diff_embeddings = han_transposed-hun_transposed
    print('Diff embedding: ', diff_embeddings)
    
    # normalizing the features
    embeddings_standarized = StandardScaler().fit_transform(han_transposed) 

    print(np.mean(embeddings_standarized), np.std(embeddings_standarized))

    emb_feat_cols = ['feature'+str(i) for i in range(embeddings_standarized.shape[1])]

    normalised_emb = pd.DataFrame(embeddings_standarized, columns=emb_feat_cols)
    print(normalised_emb)

    pca_emb = PCA(n_components=10)
    principalComponents_emb = pca_emb.fit_transform(embeddings_standarized)

    principal_emb_Df = pd.DataFrame(data = principalComponents_emb, columns = ['principal component 1', 'principal component 2', 'prinicpal component 3', 
    'principal component 4', 'principal component 5', 'principal component 6', '7', '8', '9', '10'])
    print(principal_emb_Df.tail())

    print('Explained variation per principal component: {}'.format(pca_emb.explained_variance_ratio_))

    plot_bar(pca_emb)
    #plot_scatter(principal_emb_Df)

    #hvorfor så mange prikker? 
    #få setninger 
    #hva med å plotte hun og han OG forskjell i hver sin farge?
    
