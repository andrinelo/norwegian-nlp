
from datasets import load_dataset
import json

def create_swapped_word_mapping(): 
    words_to_be_swapped = ['Han', 'Ham', 'Hans', 'Gutt', 'Gutten', 'Gutter', 'Guttene', 'Mann', 'Menn', 'Mennene', 'Herrene', 'Herrer', 'Herr']
    female_words = ['Hun', 'Hun', 'Hennes', 'Jente', 'Jenta', 'Jenter', 'Jentene', 'Kvinne', 'Kvinner', 'Kvinnene', 'Damene', 'Damer', 'Fru']
    
    swapped_dict = {}
    for i in range(len(words_to_be_swapped)):
        swapped_dict[words_to_be_swapped[i]] = female_words[i]
        swapped_dict[words_to_be_swapped[i].lower()] = female_words[i].lower()
        swapped_dict[words_to_be_swapped[i].upper()] = female_words[i].upper()
    return swapped_dict

def access_NCC(access_token): 
    data = load_dataset('NbAiLab/NCC', streaming=True, use_auth_token=access_token)
    #use training set only instead of
    training_data = data['train']
    return training_data

def gender_swap_dataset(data, filename, swapped_dict): 
    for object in data:
        text = object['text']

        for male_word, female_word in swapped_dict.items(): 
            swapped_text = text.replace(male_word, female_word)
            text = swapped_text
            
        object['text'] = text
        
        with open(filename, 'a') as json_file: 
            json.dump(object, json_file, indent=4, separators=(',',': '))

if __name__ == '__main__': 

    """
    Insert your own generated access token from HuggingFace here
    """
    access_token = ''

    """
    Insert your own path for new json-file with gender swapped json objects
    """
    filename = ''
    

    swapped_dict = create_swapped_word_mapping()
    training_data = access_NCC(access_token)

    """
    iterates through the WHOLE training part of the dataset, 
    gender swappes a set of words from swapped_dict 
    and dumps the new json object in a file named filename
    """
    gender_swap_dataset(training_data, filename, swapped_dict)
