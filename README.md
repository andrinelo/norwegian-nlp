# master-thesis
## Count pronouns in Norwegian Language Model NorBERT
NorBERT is trained on 
* Norsk Aviskorpus; 1.7 billion words;
* Bokmål Wikipedia; 160 million words;
* Nynorsk Wikipedia; 40 million words;

The corpus files are excluded from the code due to size and easy availability online.

To count the number of pronouns in Norsk Aviskorpus:
1. Download Norsk Aviskorpus from https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-4/. 
2. Unpack .tar.gz and .gz files in folder ../master-thesis\experiments\pronoun_count\NorBERT\Norsk_aviskorpus
3. Add path to files in "rootdir" in main()
4. Run experiments\pronoun_count\pronoun_count_norsk_aviskorpus.py


To count the number of pronouns in Wikipedia:
1. Download Bokmål Wikipedia (https://dumps.wikimedia.org/nowiki/latest/) and Nynorsk Wikipedia (https://dumps.wikimedia.org/nnwiki/latest/) dumps with segment wiki (https://github.com/RaRe-Technologies/gensim/blob/master/gensim/scripts/segment_wiki.py)
2. Replace the argument in pronoun_count/pronoun_count_in_wikipedia.py of the functions in main() with the path to your wiki-dump-jsonfile
3. Move to experiments\pronoun_count and run pronoun_count_in_wikipedia.py 
