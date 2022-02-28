"""
We generate vectors for each of the descriptions and for each fictive company name (i.e. a
male or female name, followed by “Aktiebolag”)

For the contextualized language models (ELMo and BERT), we generate vectors for each
description and each fictive name. In the case of ELMo we take the average over the three LSTM
layers, and for BERT we use the output embedding for the [CLS] token for each of the input sequences.

"""

#Calculating the distance from BERT embeddings of a text and a word/two texts

from lib2to3.pgen2 import token
from venv import create
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)
from transformers import logging
logging.set_verbosity_error()

import matplotlib.pyplot as plt
#% matplotlib inline

def tokenize_text(sentence, model): 
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(model)

    seperated_sentence = sentence.replace('.', '.[SEP]')

    marked_text = "[CLS] " + seperated_sentence

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    return tokenized_text


def get_emb_hidn_sates(tokenized_text, model): 

    print('Tokenized text: ', tokenized_text)

    tokenizer = BertTokenizer.from_pretrained(model)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Display the words with their indeces.
    for tup in zip(tokenized_text, indexed_tokens):
        print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    # Mark each of the tokens as belonging to sentence sentence 0, sentence 1 or so.
    segments_ids = []
    id_counter = 0
    for token in tokenized_text: 
        segments_ids.append(id_counter)
        if token == '[SEP]':
            id_counter += 1

    #segments_ids = [1] * len(tokenized_text)

    print('segments IDs:', segments_ids)


    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(model,
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    #Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
    layer_i = 0

    print ("Number of batches:", len(hidden_states[layer_i]))
    batch_i = 0

    print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
    token_i = 0

    print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

    """
    # For the 5th token in our sentence, select its feature values from layer 5.
    token_i = 4
    layer_i = 5
    vec = hidden_states[layer_i][batch_i][token_i]
    """

    # `hidden_states` is a Python list.
    print('      Type of hidden_states: ', type(hidden_states))

    # Each layer in the list is a torch tensor.
    print('Tensor shape for each layer: ', hidden_states[0].size())


    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)

    print(token_embeddings.size())


    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    print(token_embeddings.size())

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)

    print(token_embeddings.size())


    return token_embeddings, hidden_states

def create_word_vectors(tokenized_text, model):

    token_embeddings, hidden_states = get_emb_hidn_sates(tokenized_text, model)

    # Stores the token vectors, with shape [22 x 768]
    token_vectors = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.
    
    #create token embeddings
    # For each token in the sentence...
    for token in token_embeddings:

        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vectors.append(sum_vec)

    print ('Shape is: %d x %d' % (len(token_vectors), len(token_vectors[0])))

    # `hidden_states` has shape [13 x 1 x 22 x 768]

    print("gift (ektefelle)", str(token_vectors[4][:5]))
    print("gift (forgiftet) ", str(token_vectors[9][:5]))
    print("gift (ektefelle)", str(token_vectors[16][:5]))

    return token_vectors

def create_sentence_embedding(tokenized_text, model):

    token_embeddings, hidden_states = get_emb_hidn_sates(tokenized_text, model)

    #create sentence vectors
    # `token_vecs` is a tensor with shape [number of tokens x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    print ("Our final sentence embedding vector of shape:", sentence_embedding.size())

    #print("Sentence embedding: ", sentence_embedding)

    return sentence_embedding 


def cosine_similarity(sentence1, sentence2, model): 

    tokenized_text1 = tokenize_text(sentence1, model)
    tokenized_text2 = tokenize_text(sentence2, model)

    sentence_embedding1 = create_sentence_embedding(tokenized_text1, model)
    sentence_embedding2 = create_sentence_embedding(tokenized_text2, model)
    #token_vectors = create_word_vectors(tokenized_text, model)

    #create mapping from word to vector
    for i, token_str in enumerate(tokenized_text1):
        print(i, token_str)

    cosine_similarity = 1 - cosine(sentence_embedding1, sentence_embedding2)
    #sim = 1 - cosine(sentence_embedding1, sentence_embedding1)

    #print('Sim: ', sim)
    #print('Cosine similarity between "{}" and "{}": {}'.format(sentence1, sentence2, cosine_similarity))

    return cosine_similarity

    # Calculate the cosine similarity between the word bank 
    # in "bank robber" vs "river bank" (different meanings).
    #diff_bank = 1 - cosine(sentence_embedding, sentence_embedding)

    # Calculate the cosine similarity between the word bank
    # in "bank robber" vs "bank vault" (same meaning).
    #same_bank = 1 - cosine(token_vectors[4], token_vectors[16])

    #print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)
    #print('Vector similarity for *different* meanings:  %.2f' % diff_bank)


if __name__ == '__main__':

    NorBERT = 'ltgoslo/norbert'
    NB_BERT = 'NbAiLab/nb-bert-base'
    mBERT = 'bert-base-multilingual-cased'

    model = mBERT
    female_sentence = "Hun er en ambisiøs og dyktig kvinne som har oppnådd mye i sine prosjekter. Hun søker om finansiering til prosjektet som hun har startet sammen med sin kollega gjennom mange år, Anne."
    male_sentence = "Han er en ambisiøs og dyktig mann som har oppnådd mye i sine prosjekter. Han søker om finansiering til prosjektet som han har startet sammen med sin kollega gjennom mange år, Lars."
    sentence = 'Prosjektet har høy grad av innovasjon og gjennomføringsevne.'

    cosine_similarity_female = cosine_similarity(female_sentence, sentence, model)
    cosine_similarity_male = cosine_similarity(male_sentence, sentence, model)

    print('Cosine similarity between "{}" and "{}": {} \n'.format(female_sentence, sentence, cosine_similarity_female))
    print('Cosine similarity between "{}" and "{}": {}'.format(male_sentence, sentence, cosine_similarity_male))

