# master-thesis
## Count pronouns in Norwegian Language Model NorBERT
NorBERT is trained on 
* Norsk Aviskorpus; 1.7 billion words;
* Bokmål Wikipedia; 160 million words;
* Nynorsk Wikipedia; 40 million words;

The corpus files are excluded from the code due to size and easy availability online. We collected this data on the 20th of January. 

To count the number of pronouns in Norsk Aviskorpus:
1. Download [Norsk Aviskorpus](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-4/) 
2. Unzip .tar.gz and .gz files
3. Replace the variable in "rootdir" in main() with the path to your Aviskorpus data
4. Run experiments/pronoun_count/pronoun_count_norsk_aviskorpus.py


To count the number of pronouns in Wikipedia:
1. Download [Bokmål Wikipedia](https://dumps.wikimedia.org/nowiki/latest/) and [Nynorsk Wikipedia](https://dumps.wikimedia.org/nnwiki/latest/) dumps with [segment wiki](https://github.com/RaRe-Technologies/gensim/blob/master/gensim/scripts/segment_wiki.py)
2. Replace the argument in pronoun_count/pronoun_count_in_wikipedia.py of the functions in main() with the path to your wiki-dump-jsonfile
3. Run experiments/pronoun_count/pronoun_count_in_wikipedia.py 

Our experiment was performed 20th of January and resulted in this count for the pronouns "han", "ham", "hun", "ho", "henne"
|  | NO-Wikipedia        | NN-Wikipedia           | Aviskorpus  |
| -------------| ------------- |:-------------:| -----:|
| Female pronouns  | 254 752  | 62 667 | 2 304 084 |
| Male pronouns   | 918 999      | 239 107     |   7 539 723 |

This is the result for the count of the gendered words "mann", "kvinne", "gutt", "gut", "jente", "herre", "dame"
|  | NO-Wikipedia        | NN-Wikipedia           | Aviskorpus  |
| -------------| ------------- |:-------------:| -----:|
| Female pronouns  | 8 813  | 1 569 | 259 924 |
| Male pronouns   | 18 744      | 3 328     |   490 919 |
