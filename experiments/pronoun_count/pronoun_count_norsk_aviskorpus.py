
from encodings import utf_8
import codecs
from logging import root
import os


#The corpus found and downloaded from https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-4/. 
#.tar.gz and .gz files unzipped in folder named in master-thesis\NorBERT\Norsk_aviskorpus

def get_text_from_file(file):
    #Extract all text from file and return a text without \n and \r
    print('=== Starting read from file ===')
    with codecs.open(file, 'r', encoding='ISO-8859-1') as f:
        print('Opening file...')
        print('Reading file...')
        lines = f.readlines()
        new_list = []
        for line in lines: 
            new_list.append(line.replace("\n"," ").replace('\r', ' '))
        stripped_text = ''.join(new_list)
        return stripped_text

def count_pronouns(string):
    #Iterate through a string (text) and count the pronouns. Keep track of global counting 
    # in pronoun_count_dictionary and return the number of pronouns and other words counted
    print('=== Starting count of pronouns ===')
    count_all_other_words = 0
    count_all_pronouns = 0
    for word in string.split(' '): 
        if word in pronouns: 
            pronoun_count_dictionary[word] += 1
            count_all_pronouns += 1
        else: 
            count_all_other_words += 1
    #print('All pronouns counted. Resulted in ', pronoun_count_dictionary)
    #print('That is ', count_all_pronouns, 'pronouns and ', count_all_other_words, 'other words')
    return count_all_pronouns, count_all_other_words

def group_gender_pronouns(dict): 
    #Grop the male and female pronouns from pronoun_count_dictionary and return a string with the 
    # summed number of each gender (formatted as a string to be printed)
    male_pronouns = dict['han'] + dict['ham']
    female_pronouns = dict['hun'] + dict['henne'] + dict['ho']
    return ' [Male pronouns: ' + str(male_pronouns) + ', female pronouns ' + str(female_pronouns) + ']'

def iterate_files(rootdir, count_all_other_words_total, count_all_pronouns_total): 
    #Iterate directory of files from root directory and do pronoun counting for each file
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            text = get_text_from_file((os.path.join(subdir, file)))
            count_all_pronouns, count_all_other_words = count_pronouns(text)
            count_all_other_words_total += count_all_other_words
            count_all_pronouns_total += count_all_pronouns
    return count_all_other_words_total, count_all_pronouns_total

if __name__ == '__main__':

    #Define root directory
    rootdir =  r'C:\Users\regineru\Desktop\code\Fordypningsoppgave\NorBERT\Norsk_aviskorpus/'
    pronouns = ['han', 'ham', 'hun', 'ho', 'henne']
    
    pronoun_count_dictionary = {i: 0 for i in pronouns}
    count_all_other_words_total = 0
    count_all_pronouns_total = 0

    #Run
    count_all_other_words_total, count_all_pronouns_total = iterate_files(rootdir, count_all_other_words_total, count_all_pronouns_total)
    
    #Printing results
    print('==================TOTALT==================')
    print('All pronouns counted. Resulted in ', pronoun_count_dictionary)
    print('That is ', count_all_pronouns_total, 'pronouns and ', count_all_other_words_total, 'other words')

    gendered_pronouns_text_total = group_gender_pronouns(pronoun_count_dictionary)
    print('Grouped by gender: ' + gendered_pronouns_text_total)