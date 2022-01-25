from professions_from_ssb import all_professions
import re

"""
FAGOPERATØR (TREFORDELINGINGUSRI), FAGOPERATØR (CELLULOSE), FAGOPERATØR (PAPIRPRODUKSJON), 
FAGOPERATØR (SMELTEVERK)---> FAGOPERATØR
"""

all_professions = re.sub(r'\(.*\)', '', all_professions)

"""
Make into list of professions and lower caps
"""
professions = [word.lower() for word in (all_professions.split('\n')) if word]


"""
Remove duplicates
Number of proffessions goes from 6877 to 4553
"""
professions = list(set(professions))
