
from encodings import utf_8
import codecs
from logging import root
import os


def get_text_from_file_2_3(file):
    print('=== Starting read from file ===')
    with codecs.open(file, 'r', encoding='ISO-8859-1') as f:
        print('Opening file...')
        #firstline = f.readline().rstrip()
        #print(firstline)
        print('Reading file...')
        lines = f.readlines()
        new_list = []
        for line in lines: 
            new_list.append(line.replace("\n"," ").replace('\r', ' '))
        #print(new_list)
        stripped_text = ''.join(new_list)
        #print(stripped_text)
        stripped_list = stripped_text.split(' ')
        print(stripped_list)
        return stripped_text

def get_text_from_file(file):
    print('=== Starting read from file ===')
    with codecs.open(file, 'r', encoding='ISO-8859-1') as f:
        print('Opening file...')
        #firstline = f.readline().rstrip()
        #print(firstline)
        print('Reading file...')
        lines = f.readlines()
        new_list = []
        for line in lines: 
            new_list.append(line.replace("\n"," ").replace('\r', ' '))
        #print(new_list)
        stripped_text = ''.join(new_list)
        return stripped_text

def count_pronouns(string):
    print('=== Starting count of pronouns ===')
    count_all_other_words = 0
    count_all_pronouns = 0
    for word in string.split(' '): 
        if word in pronouns: 
            pronoun_count_dictionary[word] += 1
            count_all_pronouns += 1
        else: 
            count_all_other_words += 1
    print('All pronouns counted. Resulted in ', pronoun_count_dictionary)
    print('That is ', count_all_pronouns, 'pronouns and ', count_all_other_words, 'other words')
    return count_all_pronouns, count_all_other_words

def group_gender_pronouns(dict, filename): 
    male_pronouns = dict['han'] + dict['ham']
    female_pronouns = dict['hun'] + dict['henne'] + dict['ho']
    return 'From file ', + filename + ' [Male pronouns: ' + str(male_pronouns) + ', female pronouns ' + str(female_pronouns) + ']'

def iterate_files(rootdir): 
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print('Reading from file ' + str(os.path.join(subdir, file)))
            text = get_text_from_file(file)
            count_pronouns(text)
            gendered_pronouns_text = group_gender_pronouns(pronoun_count_dictionary, str(os.path.join(subdir, file)))
            save_to_file('pronoun_count/gendered_pronouns.txt', gendered_pronouns_text)
            yield pronoun_count_dictionary

def save_to_file(filename, text):
    print('=== Saving number of pronouns to file ===')
    write_text = text
    with open(filename, 'a') as file: 
        print('Create file...')
        file.write(write_text)
        file.write('\n')
        print('Content saved!')


#iterate_files(rootdir)

if __name__ == '__main__':
    rootdir_1 =  r'C:\Users\regineru\Desktop\code\Fordypningsoppgave\NorBERT\Norsk_aviskorpus/1'
    rootdir_2 =  r'C:\Users\regineru\Desktop\code\Fordypningsoppgave\NorBERT\Norsk_aviskorpus/2'
    rootdir_3 =  r'C:\Users\regineru\Desktop\code\Fordypningsoppgave\NorBERT\Norsk_aviskorpus/3'
    
    pronouns = ['han', 'ham', 'hun', 'ho', 'henne']
    pronoun_count_dictionary = {i: 0 for i in pronouns}

    count_all_other_words_total = 0
    count_all_pronouns_total = 0


    for subdir, dirs, files in os.walk(rootdir_1):
        for file in files:
            print('Reading from file ' + str(os.path.join(subdir, file)))
            text = get_text_from_file((os.path.join(subdir, file)))
            count_all_pronouns, count_all_other_words = count_pronouns(text)
            count_all_other_words_total += count_all_other_words
            count_all_pronouns_total += count_all_pronouns
       
    for subdir, dirs, files in os.walk(rootdir_2):
        for file in files:
            print('Reading from file ' + str(os.path.join(subdir, file)))
            text = get_text_from_file((os.path.join(subdir, file)))
            count_all_pronouns, count_all_other_words = count_pronouns(text)
            count_all_other_words_total += count_all_other_words
            count_all_pronouns_total += count_all_pronouns

    for subdir, dirs, files in os.walk(rootdir_3):
        for file in files:
            print('Reading from file ' + str(os.path.join(subdir, file)))
            text = get_text_from_file((os.path.join(subdir, file)))
            count_all_pronouns, count_all_other_words = count_pronouns(text)
            count_all_other_words_total += count_all_other_words
            count_all_pronouns_total += count_all_pronouns
    
    print('==================TOTALT==================')
    print('All pronouns counted. Resulted in ', pronoun_count_dictionary)
    print('That is ', count_all_pronouns_total, 'pronouns and ', count_all_other_words_total, 'other words')

    gendered_pronouns_text_total = group_gender_pronouns(pronoun_count_dictionary, '\n =================== \n TOTALT \n')
    save_to_file('pronoun_count/gendered_pronouns.txt', '\n \n Sum: ' + str(gendered_pronouns_text_total))