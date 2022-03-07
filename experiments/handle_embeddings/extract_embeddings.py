from create_embeddings_help_methods import *

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
def extract_all_embeddings_for_specific_word_in_multiple_sentences(list_of_sentences, model, word):
    
    all_embedding_representations = []
    for sentence in list_of_sentences: 
        print('Sentence: ', sentence)
        tokenized_text, tokens_tensor, segments_tensors = tokenize_text(sentence, model)
        token_embeddings, hidden_states = get_emb_hidden_states(tokens_tensor, segments_tensors, model)
        
        emb = create_embedding_for_specific_word_single_mention(token_embeddings, tokenized_text, word)
        all_embedding_representations.append(emb)
    
    torch_emb = torch.stack(all_embedding_representations, dim=0)
        
    print('Final size of all embeddings for specific word with single mention in multiple sentences: ', torch_emb.size())

    return torch_emb

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

    #Testing methods
    """
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
    """