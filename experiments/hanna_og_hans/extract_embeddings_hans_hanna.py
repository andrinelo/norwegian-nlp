from fileinput import filename
import sys
sys.path.insert(1, 'experiments/handle_embeddings')
from extract_embeddings import *

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def read_txt_file_to_text(filename): 
    with open(filename, encoding='utf-8') as f: 
        text = f.read().strip()
        return text

def convert_text_to_list(text): 
    text_list = text.split('.')
    text_list_dot = [sentence + '.' for sentence in text_list]
    return text_list_dot

def get_total_emb_sentence_context(text, model): 
    text_list_dot = convert_text_to_list(text)
    total = []
    for sentence in text_list_dot:
        emb = extract_total_embedding_from_text(sentence, model)
        if emb is not None: 
            total.append(emb)
    torch_emb = torch.stack(total, dim=0)
    mean_emb = torch.mean(torch_emb, dim=0)
    #print('Size of total embedding as average of total of all sentences: ', mean_emb.size())
    return mean_emb

def get_avg_han_hun(text, model, hun_han): 
    text_list_dot = convert_text_to_list(text)
    total = []
    for sentence in text_list_dot:
        emb = extract_average_embedding_for_specific_word_multiple_mentions_in_sentence(sentence, model, hun_han)
        if emb is not None: 
            total.append(emb)
    torch_emb = torch.stack(total, dim=0)
    mean_emb = torch.mean(torch_emb, dim=0)
    #print('Size of average embedding for all han/hun mentions: ', mean_emb.size())
    
    return mean_emb


def run(model, model_name, hanna_text, hans_text): 

    print('Running script for ', model_name)


    type_of_embeddings = 'Sentence_Embeddings'
    print('Extracting total embeddings for Hanna...')
    hanna_emb = get_total_emb_sentence_context(hanna_text, model)
    print('Extracting total embeddings for Hans...')
    hans_emb = get_total_emb_sentence_context(hans_text, model)

    np.savetxt('debiasing/remove_gender_subspace/data/{}_hans_SA.txt'.format(model_name), hans_emb.numpy())
    np.savetxt('debiasing/remove_gender_subspace/data/{}_hanna_SA.txt'.format(model_name), hanna_emb.numpy())


    type_of_embeddings = 'Word_Embeddings_HAN_HUN'
    print('Extracting average embeddings for HUN in Hanna text...')
    hanna_avg = get_avg_han_hun(hanna_text, model, 'hun')
    print('Extracting average embeddings for HAN in Hans text...')
    hans_avg = get_avg_han_hun(hans_text, model, 'han')

    np.savetxt('debiasing/remove_gender_subspace/data/{}_hans_TWA.txt'.format(model_name), hans_avg.numpy())
    np.savetxt('debiasing/remove_gender_subspace/data/{}_hanna_TWA.txt'.format(model_name), hanna_avg.numpy())

    
if __name__ == '__main__': 

    NorBERT = 'ltgoslo/norbert'
    NB_BERT = 'NbAiLab/nb-bert-base'
    mBERT = 'bert-base-multilingual-cased'

    model_list = [NorBERT, NB_BERT, mBERT]
    model_name = ['NorBERT', 'NB-BERT', 'mBERT']
    
    hanna_text = read_txt_file_to_text('data_sets/hanna.txt')
    hans_text = read_txt_file_to_text('data_sets/hans.txt')

    for i in range(len(model_list)):
        run(model_list[i], model_name[i], hanna_text, hans_text)