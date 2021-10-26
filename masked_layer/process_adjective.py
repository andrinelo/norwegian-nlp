from adjective_from_wikipedia import all_adjective


"""
Make into list of adjectives and lower caps
"""
adjectives = [word.lower() for word in (all_adjective.split('\n')) if word]


print(len(all_adjective)) #15568 adjektiv
