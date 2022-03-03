from ftplib import all_errors
from statistics import mean
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)
from transformers import logging
logging.set_verbosity_error()

import matplotlib.pyplot as plt


def tokenize_text(sentence, model): 
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(model)

    seperated_sentence = sentence.replace('.', '.[SEP]')

    marked_text = "[CLS] " + seperated_sentence

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    #create mapping from word to vector
    """
    for i, token_str in enumerate(tokenized_text):
        print(i, token_str)
    """
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Display the words with their indeces.
    """
    for tup in zip(tokenized_text, indexed_tokens):
        print('{:<12} {:>6,}'.format(tup[0], tup[1]))
    """
    # Mark each of the tokens as belonging to sentence sentence 0, sentence 1 or so.
    segments_ids = []
    id_counter = 0
    for token in tokenized_text: 
        segments_ids.append(id_counter)
        if token == '[SEP]':
            id_counter += 1

    #segments_ids = [1] * len(tokenized_text)


    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_emb_hidden_states(tokens_tensor, segments_tensors, model):
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

    """
    print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
    layer_i = 0

    print ("Number of batches:", len(hidden_states[layer_i]))
    batch_i = 0

    print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
    token_i = 0

    print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

    # `hidden_states` is a Python list.
    print('      Type of hidden_states: ', type(hidden_states))

    # Each layer in the list is a torch tensor.
    print('Tensor shape for each layer: ', hidden_states[0].size())
    """

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)

    return token_embeddings, hidden_states

def convert_all_token_embeddings_to_token_vectors(token_embeddings): 
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

    torch_emb = torch.stack(token_vectors, dim=0)
    #print ("Our final sentence embedding vector of shape:", torch_emb.size())
    return torch_emb


def create_sentence_embedding_from_hidden_sates(hidden_states):
    #create sentence vectors
    # `token_vecs` is a tensor with shape [number of tokens x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    #print ("Our final sentence embedding vector of shape:", sentence_embedding.size())
    #print("Sentence embedding: ", sentence_embedding)
    return sentence_embedding 


def create_embedding_for_specific_word_single_mention(token_embeddings, tokenized_text, word): 
    vec = convert_all_token_embeddings_to_token_vectors(token_embeddings)

    # Find the position of gendered word in in list of tokens
    word_index = tokenized_text.index(word) # = dét ordet som er i alle setningne i form av ulike kontekster
    # Get the embedding for bank
    #word_embedding = list_token_embeddings[word_index]
    
    word_embedding = vec[word_index]
    return word_embedding

def create_embeddings_for_all_representations_of_a_word_multiple_mentions(token_embeddings, tokenized_text, word): 
    vec = convert_all_token_embeddings_to_token_vectors(token_embeddings)
    # Getting embeddings for the target word in all given contexts
    target_word_embeddings = []
        
    for w in tokenized_text: 
        if w.replace('.', '') == word: 
            # Find the position of gendered word in in list of tokens
            word_index = tokenized_text.index(word) # = dét ordet som er i alle setningne i form av ulike kontekster
            # Get the embedding for bank
            word_embedding = vec[word_index]

            target_word_embeddings.append(word_embedding) #= embeddings for "bank" i kontekst av de ulike setningene
    
    torch_emb = torch.stack(target_word_embeddings, dim=0)
    #print ("Our final sentence embedding vector of shape:",  torch_emb.size())

    return torch_emb

def cosine_similarity(embedding1, embedding2): 

    cosine_similarity = 1 - cosine(embedding1, embedding2)
    #sim = 1 - cosine(sentence_embedding1, sentence_embedding1)

    #print('Sim: ', sim)
    #print('Cosine similarity between "{}" and "{}": {}'.format(sentence1, sentence2, cosine_similarity))

    return cosine_similarity




"""
The following functions should be moved to higher level file
"""



# Extract the sentence embedding for a whole input sentence
def extract_total_embedding_from_text(text, model): 
    tokenized_text, tokens_tensor, segments_tensors = tokenize_text(text, model)
    token_embeddings, hidden_states = get_emb_hidden_states(tokens_tensor, segments_tensors, model)
    sentence_embedding = create_sentence_embedding_from_hidden_sates(hidden_states)
    print('Final size of total sentence embedding from text: ', sentence_embedding.size())
    return sentence_embedding

# Extract the embeddings for all words in a sentence
def extract_word_embeddings_for_all_tokens_from_text(text, model): 
    tokenized_text, tokens_tensor, segments_tensors = tokenize_text(text, model)
    token_embeddings, hidden_states = get_emb_hidden_states(tokens_tensor, segments_tensors, model)

    token_vectors = convert_all_token_embeddings_to_token_vectors(token_embeddings)
    print('Final size of embedding for all tokens in a text: ', token_vectors.size())
    return token_vectors

def extract_embedding_for_specific_word_in_text_single_mention(text, model, word): 
    tokenized_text, tokens_tensor, segments_tensors = tokenize_text(text, model)
    token_embeddings, hidden_states = get_emb_hidden_states(tokens_tensor, segments_tensors, model)
    emb = create_embedding_for_specific_word_single_mention(token_embeddings, tokenized_text, word)
    
    print('Final size of embedding for specific word in text with single mention: ', emb.size())
    return emb

def extract_all_embeddings_for_specific_word_in_text_multiple_mentions(text_with_multiple_mentions, model, word): 
    tokenized_text, tokens_tensor, segments_tensors = tokenize_text(text_with_multiple_mentions, model)
    token_embeddings, hidden_states = get_emb_hidden_states(tokens_tensor, segments_tensors, model)
    
    all_embedding_representations = create_embeddings_for_all_representations_of_a_word_multiple_mentions(token_embeddings, tokenized_text, word)
    print('Final size of embedding for specific word in text with multiple mentions: ', all_embedding_representations.size())
    return all_embedding_representations

# Extract the average word embedding of a specific word from a text og list of texts that has multiple mentions in EACH
def extract_average_embedding_for_specific_word_multiple_mentions_in_sentence(text_with_multiple_mentions, model, word):
    tokenized_text, tokens_tensor, segments_tensors = tokenize_text(text_with_multiple_mentions, model)
    token_embeddings, hidden_states = get_emb_hidden_states(tokens_tensor, segments_tensors, model)
    
    all_embedding_representations = create_embeddings_for_all_representations_of_a_word_multiple_mentions(token_embeddings, tokenized_text, word)
    mean_emb = torch.mean(all_embedding_representations, dim=0)
    
    print('Final size of mean embedding for specific word with multiple mentions: ', mean_emb.size())
    return mean_emb

#To use for finding "hun" and "han" in PCA
def extract_average_embedding_for_specific_word_in_multiple_sentences(list_of_sentences, model, word):
    
    all_embedding_representations = []
    for sentence in list_of_sentences: 
        tokenized_text, tokens_tensor, segments_tensors = tokenize_text(sentence, model)
        token_embeddings, hidden_states = get_emb_hidden_states(tokens_tensor, segments_tensors, model)
        
        emb = create_embedding_for_specific_word_single_mention(token_embeddings, tokenized_text, word)
        all_embedding_representations.append(emb)
    
    torch_emb = torch.stack(all_embedding_representations, dim=0)
        
    mean_emb = torch.mean(torch_emb, dim=0)
    print('Final size of mean embedding for specific word with single mention in multiple sentences: ', mean_emb.size())

    return mean_emb

def cosine_similarity(embedding1, embedding2): 

    cosine_similarity = 1 - cosine(embedding1, embedding2)
    #sim = 1 - cosine(sentence_embedding1, sentence_embedding1)

    #print('Sim: ', sim)
    #print('Cosine similarity between "{}" and "{}": {}'.format(sentence1, sentence2, cosine_similarity))

    return cosine_similarity


if __name__ == '__main__': 

    model = 'ltgoslo/norbert'
    text_single = "ikke gi meg bank."
    text_multiple = 'gir du meg bank så gir jeg deg bank også før noen andre gir deg bank.'
    list_of_sentences = ["jeg er en jente jeg.", 'en snill jente faktisk.', 'jeg ser en jente.']

 
    emb_total = extract_total_embedding_from_text(text_single, model)

    
    emb_all = extract_word_embeddings_for_all_tokens_from_text(text_single, model)

    emb_word = extract_embedding_for_specific_word_in_text_single_mention(text_single, model, 'bank')
    emb_word2 = extract_all_embeddings_for_specific_word_in_text_multiple_mentions(text_multiple, model, 'bank')

    emb_avg = extract_average_embedding_for_specific_word_in_multiple_sentences(list_of_sentences, model, 'jente')
    emb_avg2 = extract_average_embedding_for_specific_word_multiple_mentions_in_sentence(text_multiple, model, 'bank')


    cos1 = cosine_similarity(emb_total, emb_total)
    cos2 = cosine_similarity(emb_all[0], emb_all[0])
    cos3 = cosine_similarity(emb_all[0], emb_word2[0])

    cos4 = cosine_similarity(emb_word, emb_avg)
    cos5 = cosine_similarity(emb_avg2, emb_avg)
  

    print('Cos = 1: ', cos1)
    print('Cos = 1: ', cos2)
    print('Cos =/= 1: ', cos3)
    print('Cos =/= 1: ', cos4)
    print('Cos =/= 1: ', cos5)
