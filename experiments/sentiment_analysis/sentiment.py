#!/usr/bin/python
import plotly.express as px
from builtins import print
import pandas as pd
from transformers import BertForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
# logging.basicConfig(level=logging.INFO)
from transformers import logging
logging.set_verbosity_error()


def calculateLogits(sentence):
    encoding = tokenizer(
        [sentence], return_tensors="pt", padding=True, truncation=True, max_length=256
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    outputs = model(input_ids, attention_mask=attention_mask)
    return outputs.logits.clone().detach()[0].numpy()

# Apply calculateLogits function to hanna and hans sentences column to generate a new columns called logit column


def calculateLogitsDataframe():
    df['Hanna_Logit'] = df['Hanna'].apply(calculateLogits)
    df['Hans_Logit'] = df['Hans'].apply(calculateLogits)

# calculateLogits("Hun viste alltid stor interesse for det faren drev med.")


def softMax(tuple):
    m = softmax(tuple)
    return m


def softMaxDataframe():
    df['Hanna_Softmax'] = df['Hanna_Logit'].apply(softMax)
    df['Hans_Softmax'] = df['Hans_Logit'].apply(softMax)


def predictSentimentWithArgmax(tuple):
    m = np.argmax(tuple)
    if(m == 0):
        return "Positive"
    else:
        return "Negative"


def predictSentimentDataframe():
    df['Hanna_Sentiment'] = df['Hanna_Logit'].apply(
        predictSentimentWithArgmax)
    df['Hans_Sentiment'] = df['Hans_Logit'].apply(
        predictSentimentWithArgmax)


def findPositiveDifference(x, y):
    m = x[0]-y[0]
    return m


def findNegativeDifference(x, y):
    m = x[1]-y[1]
    return m


"""
Test code
tuple1, tuple2 = [0.034401156, 0.1421119], [-0.007320922, 0.14639825]
hans, hans2 = [-0.525, 0.1], [-0.100, 0.05]
print(findDifference(tuple1, tuple2),
    "Should be a positive number indicating Hanna is more positive")
print(findDifference(hans, hans2),
    "Should be a negative number indicating Hans is more positive")

"""


def findDifferenceDataframe():
    df["Diff pos sentiment"] = df.apply(lambda x: findPositiveDifference(
        x["Hanna_Logit"], x["Hans_Logit"]), axis=1)
    df["Diff neg sentiment"] = df.apply(lambda x: findNegativeDifference(
        x["Hanna_Logit"], x["Hans_Logit"]), axis=1)


"""
For hver gang [pos sentiment] er positivt så pluss en på Hanna oppfattes mer positiv
For hver gang [pos sentiment] er negativ så pluss en på at Hans oppfattes mer positiv

For hver gang [neg sentiment] er positivt så pluss en på Hanna oppfattes mer negativ
For hver gang [neg sentiment] er negativ så pluss en på Hans oppfattes mer negativ
"""


def getPositiveSentiment(diff):
    if diff > 0:
        return "Hanna more positive"
    elif diff < 0:
        return "Hans more positive"
    else:
        return "Neutral"


def getNegativeSentiment(diff):
    if diff > 0:
        return "Hanna more negative"
    elif diff < 0:
        return "Hans more negative"
    else:
        return "Neutral"


def getSentimentDataframe():
    df["Pos polarity"] = df["Diff pos sentiment"].apply(
        getPositiveSentiment)
    df["Neg polarity"] = df["Diff neg sentiment"].apply(
        getNegativeSentiment)


if __name__ == '__main__':
    modelname = "ltgoslo/norbert"  # NorBert, mBERT and NB-BERT as well
    model = BertForSequenceClassification.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)

    print("............................")
    print("............................")
    print("............................")
    print("............................")

    model.eval()

    df = pd.read_csv("experiments/sentiment_analysis/hanna_hans.csv")

    calculateLogitsDataframe()
    print(df.head())

    softMaxDataframe()
    print(df.head())

    predictSentimentDataframe()
    print(df.head())

    findDifferenceDataframe()
    print(df.head())

    getSentimentDataframe()
    print(df.head())

    pie_frame_positive = df["Pos polarity"].value_counts().rename_axis(
        "Who is more positive").to_frame("Total sentences pos")

    pie_frame_negative = df["Neg polarity"].value_counts().rename_axis(
        "Who is more negative").to_frame("Total sentences neg")

    """ 
    Save the results to CSV files
    """
    # df.to_csv("hanna_hans_analyze.csv")
    # pie_frame_positive.to_csv("who_is_more_positive.csv")
    # pie_frame_negative.to_csv("who_is_more_negative.csv")

    """
    Should extend to include arguments that run the program on perfered BERT model

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--model",
        "-m",
        help="Path to a BERT model: ltgoslo/norbert, NbAiLab/nb-bert-base or bert-base-multilingual-cased are possible options)",
        required=True,
    )

    args = parser.parse_args(arg)

    modelname = args.model
    """
