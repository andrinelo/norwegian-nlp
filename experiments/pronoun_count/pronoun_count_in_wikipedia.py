from gensim import utils
import json


"""
Wikipedia dumps are red using segment wiki (https://github.com/RaRe-Technologies/gensim/blob/master/gensim/scripts/segment_wiki.py),
bokmål wikipedia was downloaded from https://dumps.wikimedia.org/nowiki/latest/ and nynorsk wikipedia was downloaded from
https://dumps.wikimedia.org/nnwiki/latest/
"""

pronouns = ['han', 'ham', 'hun', 'ho', 'henne']
pronoun_count_dictionary = {i: 0 for i in pronouns}
print(pronoun_count_dictionary)

def get_text_from_zipped_json_and_count_in_titles(jsonfile):
    #Specify which wikipedia that should be red here
    with utils.open(jsonfile, 'rb') as f:
        count_all_other_words = 0
        count_all_pronouns = 0
        for line in f:
             # decode each JSON line into a Python dictionary object
            article = json.loads(line)
            
             # each article has a "title" and a list of "section_titles" and "section_texts".
            for word in article['title'].split(' '):
                    if word in pronouns:
                        pronoun_count_dictionary[word] += 1
                        count_all_pronouns += 1
                    else: 
                        count_all_other_words += 1
            for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                print("Section title: %s" % section_title)
                # print("Section text: %s" % section_text)
                for word in section_title.split(' '):
                    if word in pronouns:
                        pronoun_count_dictionary[word] += 1
                        count_all_pronouns += 1
                    else: 
                        count_all_other_words += 1
    print('All pronouns counted. Resulted in ', pronoun_count_dictionary)
    print('That is ', count_all_pronouns, 'pronouns and ', count_all_other_words, 'other words')
    return pronoun_count_dictionary       


#Same as above just for the section texts      
def get_text_from_zipped_json_and_count_in_text(jsonfile):
    with utils.open(jsonfile, 'rb') as f:
        count_all_other_words = 0
        count_all_pronouns = 0
        for line in f:
            article = json.loads(line)
            for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                for word in section_text.split(' '):
                    if word in pronouns:
                        pronoun_count_dictionary[word] += 1
                        count_all_pronouns += 1
                    else: 
                        count_all_other_words += 1
    print('All pronouns counted. Resulted in ', pronoun_count_dictionary)
    print('That is ', count_all_pronouns, 'pronouns and ', count_all_other_words, 'other words')
    return pronoun_count_dictionary 



def main(): 
    #pronoun_count_dictionary = get_text_from_zipped_json_and_count_in_titles('nowiki-latest.json.gz')
    #pronoun_count_dictionary_text = get_text_from_zipped_json_and_count_in_text('nowiki-latest.json.gz')
    #pronoun_count_dictionary = get_text_from_zipped_json_and_count_in_titles('nnwiki-latest.json.gz')
    #pronoun_count_dictionary_text = get_text_from_zipped_json_and_count_in_text('nnwiki-latest.json.gz')
main()
