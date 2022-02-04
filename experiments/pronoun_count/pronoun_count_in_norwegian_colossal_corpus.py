
from encodings import utf_8
import codecs
from logging import root
import os
from datasets import load_dataset
import json


pronouns = ['mann', 'kvinne', 'gutt', 'gut', 'jente', 'herre', 'dame']
pronoun_count_dictionary = {i: 0 for i in pronouns}
print(pronoun_count_dictionary)


def read_data(jsonfile):

    pronoun_count_dictionary = {i: 0 for i in pronouns}
    print(pronoun_count_dictionary)
    # Specify which wikipedia that should be red here
    count_all_other_words = 0
    count_all_pronouns = 0
    for line in jsonfile:
        # decode each JSON line into a Python dictionary object
        print("==========JSONLINE IN DATA SHOULD BE OF id doc type etc...", line)
        article = json.loads(line)

        # each article has a "title" and a list of "section_titles" and "section_texts".
        for word in article['text'].split(' '):
            if word.lower() in pronouns:
                pronoun_count_dictionary[word.lower()] += 1
                count_all_pronouns += 1
            else:
                count_all_other_words += 1
    print('All pronouns counted. Resulted in ', pronoun_count_dictionary)
    print('That is ', count_all_pronouns, 'pronouns and ',
          count_all_other_words, 'other words')
    return pronoun_count_dictionary


def group_gender_pronouns(dict):
    # Grop the male and female pronouns from pronoun_count_dictionary and return a string with the
    # summed number of each gender (formatted as a string to be printed)
    # TODO Update if updating pronouns
    male_pronouns = dict['han'] + dict['ham']
    female_pronouns = dict['hun'] + dict['henne'] + dict['ho']
    return ' [Male pronouns: ' + str(male_pronouns) + ', female pronouns ' + str(female_pronouns) + ']'


if __name__ == '__main__':
    print("Get that count")
    pronoun_count_dictionary = read_data('onefile.json')
    group_gender_pronouns(pronoun_count_dictionary)
