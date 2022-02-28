import json


#pronouns = ['han', 'ham', 'hun', 'ho', 'henne']
pronouns = ['mann', 'kvinne', 'gutt', 'gut', 'jente', 'herre', 'dame']
pronoun_count_dictionary = {i: 0 for i in pronouns}
print(pronoun_count_dictionary)


def read_data(jsonfile):
    count_all_other_words = 0
    count_all_pronouns = 0
    with open(jsonfile) as f:
        for line in f:
            data = json.loads(line)
            for word in data['text'].split(' '):
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
    # Update if updating pronouns
    male_pronouns = dict['mann'] + dict['gutt'] + dict['gut'] + dict['herre']
    female_pronouns = dict['kvinne'] + dict['jente'] + dict['dame']
    print(' [Male pronouns: ' + str(male_pronouns) +
          ', female pronouns ' + str(female_pronouns) + ']')


if __name__ == '__main__':
    pronoun_count_dictionary = read_data(
        'onefile.json')
    group_gender_pronouns(pronoun_count_dictionary)
