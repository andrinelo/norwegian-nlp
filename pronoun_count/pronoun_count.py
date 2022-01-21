
from encodings import utf_8
import codecs


pronouns = ['han', 'ham', 'hun', 'ho', 'henne']
pronoun_count_dictionary = {i: 0 for i in pronouns}
print(pronoun_count_dictionary)

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
            new_list.append(line.replace("\n"," "))
        stripped_text = ''.join(new_list)
        #print(stripped_text[:1000])
        return stripped_text

def iterate_files(): 
    """
    
    """
    return

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
    return pronoun_count_dictionary

def group_gender_pronouns(dict): 
    male_pronouns = dict['han'] + dict['ham']
    female_pronouns = dict['hun'] + dict['henne'] + dict['ho']
    return {'male pronouns': male_pronouns, 'female pronouns': female_pronouns}

def main(): 

    selected_file = r'C:\Users\regineru\Desktop\code\Fordypningsoppgave\NorBERT\Norsk_aviskorpus\1\19981013-20010307'
    text = get_text_from_file(selected_file)
    pronoun_count_dictionary = count_pronouns(text)
    gendered_pronouns = group_gender_pronouns(pronoun_count_dictionary)

    print(gendered_pronouns)

main()
