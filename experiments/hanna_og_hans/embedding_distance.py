from fileinput import filename
import sys
sys.path.insert(1, 'experiments/handle_embeddings')
from extract_embeddings import *

import torch
import pandas as pd
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

def get_all_hun_han(text, model, hun_han): 
    text_list_dot = convert_text_to_list(text)
    counter = True
    for sentence in text_list_dot:
        emb = extract_all_embeddings_for_specific_word_in_text_multiple_mentions(sentence, model, hun_han)
        if emb is not None:
            if counter: 
                total = emb
                counter = False
            else: 
                torch_emb = total
                total = torch.cat((torch_emb, emb), dim=0)
    #print('Size of all embeddings for all han/hun mentions: ', total.size())
    return total

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

def calculate_cosine(emb1, emb2): 
    sim = cosine_similarity(emb1, emb2)
    return sim


def extract_sentences():
    df = pd.read_excel('data_sets/hanna_og_hans_vurderinger.xlsx', sheet_name='Ark 1')
    sentences = df['Setning'].values
    return sentences

def get_similarity(sentence_embeddings, sentences, embedding_female, embedding_male, type_of_embeddings, model_name, filename): 

    diff_list = []


    with open(filename, 'a') as file:
        file.write('\n ===== Results for model [{}] with [{}] ===== \n'.format(model_name, type_of_embeddings))
        for index in range(len(sentence_embeddings)): 
            female = calculate_cosine(sentence_embeddings[index], embedding_female)
            male = calculate_cosine(sentence_embeddings[index], embedding_male)
            diff_list.append(male-female)
            file.write('S{}'.format(index))
            file.write('Diff (M-F): {} Diff ratio (M/F): {},\n'.format(male-female, male/female))
            #print('\n Results for sentence: ', sentences[index])
            #print('Female: {}, Male: {}, Diff (M-F): {}, Diff ratio (M/F): {}'.format(female, male, male-female, round(male/female, 2)))

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

    save_plot(plt, 'experiments/hanna_og_hans/results/diff_plot_'+type_of_embeddings+'.eps')
    save_plot(plt, 'experiments/hanna_og_hans/results/diff_plot_'+type_of_embeddings+'.png')


def run(model, model_name, filename, sentences, sentence_embeddings, variable): 

    print('Running script for ', model_name)

    if variable: 
        type_of_embeddings = 'Sentence_Embeddings'
        print('Extracting total embeddings for Hanna...')
        hanna_emb = get_total_emb_sentence_context(hanna_text, model)
        print('Extracting total embeddings for Hans...')
        hans_emb = get_total_emb_sentence_context(hans_text, model)

        print('Calculating similarities...')
        diff_list = get_similarity(sentence_embeddings, sentences, hanna_emb, hans_emb, type_of_embeddings, model_name, filename)
  
    else: 
        type_of_embeddings = 'Word_Embeddings_HAN_HUN'
        print('Extracting average embeddings for HUN in Hanna text...')
        hanna_avg = get_avg_han_hun(hanna_text, model, 'hun')
        print('Extracting average embeddings for HAN in Hans text...')
        hans_avg = get_avg_han_hun(hans_text, model, 'han')

        print('Calculating similarities...')
        diff_list = get_similarity(sentence_embeddings, sentences, hanna_avg, hans_avg, type_of_embeddings, model_name, filename)

    return diff_list, type_of_embeddings
    
    
if __name__ == '__main__': 

    NorBERT = 'ltgoslo/norbert'
    NB_BERT = 'NbAiLab/nb-bert-base'
    mBERT = 'bert-base-multilingual-cased'

    model_list = [NorBERT, NB_BERT, mBERT]
    name = ['NorBERT', 'NB-BERT', 'mBERT']
    file_name = ['experiments/hanna_og_hans/results/NorBERT_results.csv', 'experiments/hanna_og_hans/results/NB-BERT_results.csv', 'experiments/hanna_og_hans/results/mBERT_results.csv']

    hanna_text = read_txt_file_to_text('data_sets/hanna.txt')
    hans_text = read_txt_file_to_text('data_sets/hans.txt')

    sentences = extract_sentences()
    print(sentences)


    diffs = []

    for i in range(len(model_list)):
        print('Extracting embeddings for test sentences...')
        sentence_embeddings = [extract_total_embedding_from_text(sentence, NorBERT) for sentence in sentences]

        diff_list, type_of_embeddings = run(model_list[i], name[i], file_name[i], sentences, sentence_embeddings, True)
        diffs.append(diff_list)

    plot(diffs[0], diffs[1], diffs[2], sentences, type_of_embeddings)