# master-thesis
## Count pronouns of corpus
NorBERT is trained on 
* Norsk Aviskorpus; 1.7 billion words;
* Bokmål Wikipedia; 160 million words;
* Nynorsk Wikipedia; 40 million words;

NB-BERT is trained on NCC

The corpus files are excluded from the code due to size and easy availability online. We collected this data on the 20th of January. 

To count the number of pronouns in Norsk Aviskorpus:
1. Download [Norsk Aviskorpus](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-4/) 
2. Unzip .tar.gz and .gz files
3. Replace the variable in "rootdir" in main() with the path to your Aviskorpus data
4. Run experiments/pronoun_count/pronoun_count_norsk_aviskorpus.py

To count the number of pronouns in Wikipedia:
1. Download [Bokmål Wikipedia](https://dumps.wikimedia.org/nowiki/latest/) and [Nynorsk Wikipedia](https://dumps.wikimedia.org/nnwiki/latest/) dumps with [segment wiki](https://github.com/RaRe-Technologies/gensim/blob/master/gensim/scripts/segment_wiki.py)
2. Replace the argument in pronoun_count/pronoun_count_in_wikipedia.py with the path to your wiki-dump-jsonfile
3. Run experiments/pronoun_count/pronoun_count_in_wikipedia.py 

To count the number of pronouns in Norwegian Colossal Corpus (NCC):
1. Clone the training set with git clone https://huggingface.co/datasets/NbAiLab/NCC
2. Create one large training file of all shards without unpacking cat NCC/data/train*.gz > onefile.json.gz
3. Unpack with gzip -d onefile.json.gz
4. Replace the argument in experiments/pronoun_count/pronount_count_in_norwegian_colossal_corpus.py with the path to your jsonfile
5. Run experiments/pronoun_count/pronount_count_in_norwegian_colossal_corpus.py

Our experiment was performed 20th of January and resulted in this count for the pronouns "han", "ham", "hun", "ho", "henne"
|  | NO-Wikipedia        | NN-Wikipedia           | Aviskorpus  |NCC  |
| -------------| ------------- |:-------------:| -----:|-----:|
| Female pronouns  | 254 752  | 62 667 | 2 304 084 |7 216 408|
| Male pronouns   | 918 999      | 239 107     |   7 539 723 |23 151 190|

This is the result for the count of the gendered words "mann", "kvinne", "gutt", "gut", "jente", "herre", "dame"
|  | NO-Wikipedia        | NN-Wikipedia           | Aviskorpus  | NCC |
| -------------| ------------- |:-------------:| -----:|-----:|
| Female pronouns  | 8 813  | 1 569 | 259 924 |432 580|
| Male pronouns   | 18 744      | 3 328     |   490 919 |1 613 404|


## Principal Component Analysis of NorBERT, NB-BERT and mBERT
The sentences used can be found in experiments/pca/sample_sentences.xlsx and can be changed to other sentences/target words on the same format. 
To analyze (plot) the principal componants: 
1. Fill inn for wanted variables in __main__ function in experiments/pca/find_principal_components.py 
2. Run experiments/pca/find_principal_components.py 
3. Plots will be saved in experiments/pca/plots/

Example of plot: Top 10 principal components from 'han', 'hun', 'jente' and 'gutt' in NorBERT embeddings calculated from sentences in sheet 'hun_han_alle' and 'jente_gutt_tilfeldig'
![plot](experiments/pca/plots/NorBERT.png)

## Hanna and hans
Compare distance between Hanna and Hans descriptions and questions from survey. 
1. Run experiments/hanna_og_hans/embedding_distance.py
2. Change input variable True/False in run() in __main__ to differ between sentence embedding and han/hun embedding for texts. 

Example of plot: Difference (male-female) distance from sentence embeddings of survey questions to sentence embeddings from text

![plot](experiments/hanna_og_hans/diff_plot_Sentence_Embeddings.png)

## Masked language modelling 
Extract top gendered adjectives from Norwegian language models. 
1. run experiments/masked_adjectives/extract_top_adjectives.py to get files with top adjectives
2. run experiments/masked_adjectives/get_prediction_scores.py to get aggregated prediction scores for all adjectives
3. run experiments/masked_adjectives/plot_adjectives.py to get word cloud of top adjectives for all models 

Example of plot: Top female biased adjectives from NorBERT

![plot](experiments/masked_adjectives/word_cloud_female_NorBERT.png)

## Top Adjectives sentiment analysis

## Sentiment analysis
