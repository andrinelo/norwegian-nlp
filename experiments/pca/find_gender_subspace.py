from tkinter import N
from transformers import pipeline

from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA


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

    print('Finding average')

#get average vector of "han". switch row and cols to sum each colums and take average


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

def make_numpy(vector): 
    return np.array(vector)

def find_pca(han_vector, hun_vector):
    han = make_numpy(han_vector)
    hun = make_numpy(hun_vector)

    diffs = hun-han

    return diffs
    

if __name__ == '__main__':

    texts_han = ["han",
            "han er snekker.",
            "han er sykepleier.", 
            "I går var han der."] # = ord å sammenlikne target word med
    
    texts_hun = ["hun",
            "hun er snekker.",
            "hun er sykepleier.", 
            "I går var hun der."]

    han = get_embeddings_from_text(texts_han, 'han')
    hun = get_embeddings_from_text(texts_hun, 'hun')
    
    pca = find_pca(han, hun)
    transposed = np.transpose(pca) #funker ikke??
    print(pca.shape)

    #print("The gender difference or something is", pca)

#distances_df = pd.DataFrame(list_of_distances, columns=['text1', 'text2', 'distance'])