import pandas as pd
from transformers import BertForSequenceClassification, AutoTokenizer
import numpy as np
import logging
from transformers import logging
logging.set_verbosity_error()


def get_data(model_name):
    df_hun = pd.read_csv(
        'experiments/sentiment_analysis/data/{}_hun_adjectives.csv'.format(model_name))
    df_han = pd.read_csv(
        'experiments/sentiment_analysis/data/{}_han_adjectives.csv'.format(model_name))
    return df_hun, df_han


def create_sentences_han(adjective):
    return ('Han er ' + adjective)


def create_sentences_hun(adjective):
    return ('Hun er ' + adjective)


def calculateLogits(sentence):
    encoding = tokenizer(
        [sentence], return_tensors="pt", padding=True, truncation=True, max_length=256
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    outputs = model(input_ids, attention_mask=attention_mask)
    return outputs.logits.clone().detach()[0].numpy()


def predictSentimentWithArgmax(tuple):
    m = np.argmax(tuple)
    if(m == 0):
        return "Positive"
    else:
        return "Negative"


def create_dataframe(name):
    df_han['Complete sentence male'] = df_han['Adjektiv'].apply(
        create_sentences_han)
    df_hun['Complete sentence female'] = df_hun['Adjektiv'].apply(
        create_sentences_hun)

    df_hun['Female Logit'] = df_hun['Complete sentence female'].apply(
        calculateLogits)
    df_han['Male Logit'] = df_han['Complete sentence male'].apply(
        calculateLogits)

    df_hun['Sentiment'] = df_hun['Female Logit'].apply(
        predictSentimentWithArgmax)
    df_han['Sentiment'] = df_han['Male Logit'].apply(
        predictSentimentWithArgmax)

    df_hun.to_csv(
        'experiments/sentiment_analysis/{}/adjective_analyzed_hun.csv'.format(name))
    df_han.to_csv(
        'experiments/sentiment_analysis/{}/adjective_analyzed_han.csv'.format(name))
    print(df_hun.head())
    print(df_han.head())


if __name__ == '__main__':

    name = ['NorBERT', 'NB-BERT']
    model_list = ['ltgoslo/norbert', 'NbAiLab/nb-bert-base']

    for i in range(len(model_list)):
        print(name[i], "is the model name and",
              model_list[i], "is the hf model name")
        print("And those two should be equal")
        model = BertForSequenceClassification.from_pretrained(model_list[i])
        tokenizer = AutoTokenizer.from_pretrained(
            model_list[i], use_fast=False)
        model.eval()

        df_hun, df_han = get_data(name[i])
        create_dataframe(name[i])
