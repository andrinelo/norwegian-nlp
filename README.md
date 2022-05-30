## Detecting- and Measuring Experiments
### Count pronouns
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

Al results are written to terminal.

### Embeddings: Masked language modelling 
First, the most biased adjectives for all models are predicted:
1. Run experiments/masked_adjectives/extract_top_adjectives.py to get files with top adjectives for each of the models.
The predicted adjectives are stored in experiments/masked_adjectives/data/...

Further, the results are collected by calculating aggregated bias scores and plotting the top biased adjectives for all models. 
2. Run experiments/masked_adjectives/get_prediction_scores.py to get aggregated prediction scores for all adjectives per model.
3. Run experiments/masked_adjectives/plot_adjectives.py to get word cloud of top adjectives for all models.
Both results are stored in experiments/masked_adjectives/results/...

### Downstram Task: Hanna And Hans
First, the embeddings to be used in the experiment are extracted. 
1. Run experiments/hanna_og_hans/extract_embeddings_hans_hanna.py for all three models. Change input variable True/False in run() in __main__ to differ between sentence embedding (SA) and han/hun embedding (TWA) for texts. Embeddings are stored in experiments/hanna_og_hans/data/...

Further, the difference in distance between Hanna and Hans embeddings are calculated: 
2. Run experiments/hanna_og_hans/embedding_distance.py. The results are stored in experiments/hanna_og_hans/results/...

## Debiasing Experiments 
### Debiasing of language models by removing gender subspace
First, the embeddings to be used in the experiment are extracted. 
1. Run experiments/hanna_og_hans/extract_embeddings_hans_hanna.py for all three models. Change input variable True/False in run() in __main__ to differ between sentence embedding (SA) and han/hun embedding (TWA) for texts. 
2. Run debiasing/remove_gender_subspace/extract_embeddings_for_pca.py for all three models. Fill inn for wanted variables in the __main__ function before extracting.
Both sets of embeddings are stored in debiasing/remove_gender_subspace/data/...

Further, the embeddings are debiased through removing the gender subspace and the new distance between Hanna and Hans descriptions and questions from survey is calculated.
1. Run debiasing/remove_gender_subspace/remove_subspace.py. The results are stored in debiasing/remove_gender_subspace/results/...

### Debiasing of language models through retraining on female corpus
This experiment requires possibility to store large datasets and train complex language models. 

First, NCC corpus is gender swapped: 
1. Run debiasing/gender_swap/gender_swap_NCC.py.
2. Fine-tune NB-BERT on gender swapped corpus.
Both steps are done by The National Libraby of Norway in this thesis.

Further, both measuring experiments for embeddings are redone. 
For masked adjectives:
1. Run debiasing/gender_swap/masked_adjectives/extract_top_adjectives.py to get files with top adjectives for new model. The predicted adjectives are stored in debiasing/gender_swap/masked_adjectives/data/...
2. Run debiasing/gender_swap/masked_adjectives/get_prediction_scores.py to get aggregated prediction scores for all adjectives per model.
3. Run debiasing/gender_swap/masked_adjectives/plot_adjectives.py to get word cloud of top adjectives for all models.
Both results are stored in debiasing/gender_swap/masked_adjectives/results/...

For Hanna and Hans:
1. Run debiasing/gender_swap/hanna_og_hans/extract_embeddings_hans_hanna.py for both models. Change input variable True/False in run() in __main__ to differ between sentence embedding (SA) and han/hun embedding (TWA) for texts. Embeddings are stored in debiasing/gender_swap/hanna_og_hans/data/...
2. Run debiasing/gender_swap/hanna_og_hans/embedding_distance.py. The results are stored in debiasing/gender_swap/hanna_og_hans/results/...
