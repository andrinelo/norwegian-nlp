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
    df = pd.read_excel('experiments/hanna_og_hans/hanna_og_hans_vurderinger.xlsx', sheet_name='Ark 1')
    sentences = df['Setning'].values
    return sentences

def extract_words():
    df = pd.read_excel('experiments/hanna_og_hans/hanna_og_hans_vurderinger.xlsx', sheet_name='Ark 1')
    words = df['Ord'].values
    return words

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

    save_plot(plt, 'experiments/hanna_og_hans/diff_plot_'+type_of_embeddings+'.eps')
    save_plot(plt, 'experiments/hanna_og_hans/diff_plot_'+type_of_embeddings+'.png')


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
    file_name = ['experiments/hanna_og_hans/NorBERT_results.csv', 'experiments/hanna_og_hans/NB-BERT_results.csv', 'experiments/hanna_og_hans/mBERT_results.csv']

    hanna_text = read_txt_file_to_text('experiments\hanna_og_hans\hanna.txt')
    hans_text = read_txt_file_to_text('experiments\hanna_og_hans\hans.txt')

    sentences = extract_sentences()
    print(sentences)


    diffs = []

    for i in range(len(model_list)):
        print('Extracting embeddings for test sentences...')
        sentence_embeddings = [extract_total_embedding_from_text(sentence, NorBERT) for sentence in sentences]

        diff_list, type_of_embeddings = run(model_list[i], name[i], file_name[i], sentences, sentence_embeddings, True)
        diffs.append(diff_list)

    plot(diffs[0], diffs[1], diffs[2], sentences, type_of_embeddings)
    """
    type_of_embeddings = 'Sentence_Embeddings'
    
    norbert_all = [0.0009489655494689941, -0.0004741549491882324, 0.00046569108963012695, 0.0013622045516967773,  0.0007605552673339844, 0.005228102207183838, 0.0034021735191345215,  0.00268477201461792, 0.0009118318557739258, 0.0033776164054870605, 0.0008611083030700684, 0.00044840574264526367, 0.0005585551261901855]
    nbbert_all = [0.008219053037464619, 0.007631331216543913, 0.008047150913625956,  0.0070099295116961, 0.009760905057191849, 0.011491641402244568, 0.011944038327783346, 0.00812908262014389, 0.011035017669200897, 0.008744679624214768, 0.008929337374866009, 0.009933010675013065, 0.009675892069935799]
    mbert_all = [-0.0003962889313697815, -0.000747237354516983, -0.0006272830069065094, -0.00048665329813957214, -0.0006537213921546936,  -0.0006487146019935608, -0.0006741024553775787, -0.0004284083843231201, -0.0005349442362785339, -0.0006277821958065033, -0.0008183121681213379, -7.099658250808716e-05, -0.0004644244909286499]

    plot(norbert_all, nbbert_all, mbert_all, sentences, type_of_embeddings)

    type_of_embeddings = 'Word_Embeddings_HAN_HUN'

    norbert_avg = [0.03540682792663574, 0.020932257175445557, 0.028891682624816895, 0.035604655742645264, 0.036869049072265625, 0.05527776479721069, 0.045630455017089844, 0.04753124713897705, 0.03744065761566162, 0.04349839687347412, 0.029407620429992676, 0.03199422359466553, 0.03484904766082764]
    nbbert_avg = [0.0014400025829672813, 0.0016048075631260872, 0.001924663782119751, 0.0017376560717821121, 0.0014265403151512146, 0.001714412122964859, 0.0015875641256570816, 0.0016189496964216232, 0.0016769710928201675, 0.0015467171906493604, 0.0016932403668761253, 0.0015234649181365967, 0.0019463859498500824]
    mbert_avg = [0.01757870987057686, 0.025338388979434967, 0.03461392968893051, 0.026694869622588158, 0.028685137629508972, 0.018789853900671005, 0.023078974336385727, 0.027324534952640533, 0.022552743554115295, 0.03131864592432976, 0.01633954793214798, 0.045005571097135544, 0.04073863849043846]

    plot(norbert_avg, nbbert_avg, mbert_avg, sentences, type_of_embeddings)
    """