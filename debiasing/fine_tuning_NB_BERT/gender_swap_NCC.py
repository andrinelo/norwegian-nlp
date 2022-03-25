from nis import match
from datasets import load_dataset
import json
import re


"""
This python script
downloads NCC from hugging_face
iterates through all json objects and swaps male words with female words
dumps the new texts in a json file with your name of choice
"""


def access_NCC(access_token):
    data = load_dataset('NbAiLab/NCC', streaming=True,
                        use_auth_token=access_token)
    # use training set only
    training_data = data['train']
    return training_data


def swap_pronouns(match):
    replacements_pronouns = [('Hun', ['Han', 'Ham']), ('Hennes', ['Hans']),
                             ('Kvinner', ['Menn']), ('Fru', ['Herr'])]
    d = {w.lower(): repl.lower()
         for repl, words in replacements_pronouns for w in words}
    return d[match.group()]


def swap_pronouns_capital(match):
    replacements_pronouns = [('Hun', ['Han', 'Ham']), ('Hennes', ['Hans']),
                             ('Kvinner', ['Menn']), ('Fru', ['Herr'])]
    d = {w: repl for repl, words in replacements_pronouns for w in words}
    return d[match.group()]


def swap_gender(match):
    replacements_gender = [('Hennes', ['Hans']), ('Jente', ['Gutt', 'Gut']), ('Jenta', ['Gutten']), ('Genter', ['Gutter']),
                           ('Jentene', ['Guttene']), ('Kvinne', ['Mann']), ('Kvinnene', ['Mennene']), ('Damene', ['Herrene']), ('Damer', ['Herrer'])]
    d_gender = {w.lower(): repl.lower()
                for repl, words in replacements_gender for w in words}
    return d_gender[match.group()]


def swap_gender_capital(match):
    replacements_gender = [('Hennes', ['Hans']), ('Jente', ['Gutt', 'Gut']), ('Jenta', ['Gutten']), ('Genter', ['Gutter']),
                           ('Jentene', ['Guttene']), ('Kvinne', ['Mann']), ('Kvinnene', ['Mennene']), ('Damene', ['Herrene']), ('Damer', ['Herrer'])]
    d_gender = {w: repl for repl, words in replacements_gender for w in words}
    return d_gender[match.group()]


def gender_swap_dataset(data, filename, pronouns_dict, gender_dict):
    pronouns = {w.lower(): repl.lower()
                for repl, words in pronouns_dict for w in words}

    pronouns_capital = {w: repl for repl,
                        words in pronouns_dict for w in words}

    gender = {w.lower(): repl.lower()
              for repl, words in gender_dict for w in words}

    gender_capital = {w: repl for repl, words in gender_dict for w in words}

    for object in data:
        text = object['text']

        object['text'] = re.sub('|'.join(r'\b{0}\b'.format(
            re.escape(k)) for k in pronouns), swap_pronouns, text)
        object['text'] = re.sub('|'.join(r'\b{0}\b'.format(
            re.escape(k)) for k in pronouns_capital), swap_pronouns_capital, object['text'])

        object['text'] = re.sub('|'.join(r'{0}'.format(
            re.escape(k)) for k in gender), swap_gender, object['text'])

        object['text'] = re.sub('|'.join(r'{0}'.format(
            re.escape(k)) for k in gender_capital), swap_gender_capital, object['text'])

        with open(filename, 'a') as json_file:
            json.dump(object, json_file, indent=4, separators=(',', ': '))


if __name__ == '__main__':

    replacements_pronouns = [('Hun', ['Han', 'Ham']), ('Hennes', ['Hans']),
                             ('Kvinner', ['Menn']), ('Fru', ['Herr'])]

    replacements_gender = [('Hennes', ['Hans']), ('Jente', ['Gutt', 'Gut']), ('Jenta', ['Gutten']), ('Genter', ['Gutter']),
                           ('Jentene', ['Guttene']), ('Kvinne', ['Mann']), ('Kvinnene', ['Mennene']), ('Damene', ['Herrene']), ('Damer', ['Herrer'])]

    """
    Insert your own generated access token from HuggingFace here
    """
    access_token = ''

    """
    Insert your own path for new json-file with gender swapped json objects
    """
    filename = ''

    training_data = access_NCC(access_token)

    """
    iterates through the WHOLE training part of the dataset,
    gender swappes a set of words from swapped_dict
    and dumps the new json object in a file named filename
    """
    gender_swap_dataset(training_data, filename,
                        replacements_pronouns, replacements_gender)
