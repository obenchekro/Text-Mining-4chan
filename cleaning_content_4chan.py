import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

nltk.download()

def display_df(path, n):
    df = pd.read_json(path, orient='records')
    return df.head(n)

def display_infos_df(df):
    return df.info(), df.loc[df['Reply'] == ''].count(), df.loc[df['Reply'].duplicated()].count()

def clean_df(df):
    df.drop(df.loc[(df['Reply'] == '') | (df['Reply'] == ' ')].index, inplace=True)
    df['Reply'] = df['Reply'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', regex = True)
    df.drop_duplicates(inplace=True)
    
def tokenize(df, n):
    words = ""
    
    for quote in df['Reply']: 
        tokens = quote.split()
        tokens = [token.lower() for token in tokens if token.isalpha()]
        words += " ".join(tokens)
    
    tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
    freq = tokenizer.tokenize(words)
    return [token for token in freq if token not in nltk.corpus.stopwords.words('english')][:n]

def get_stopword(tokens):
    return FreqDist(tokens)

def count_stopword(tokens, n):
    return FreqDist(tokens).most_common(n)

def graph_stopwords(count_token, n, color):
    return count_token.plot(n, color=color)

def hist_stopwords_sorted(tokens, n, color, lw, size, width):
    counts = dict(Counter(tokens).most_common(30))
    labels, values = zip(*counts.items())

    index_sorted = np.argsort(values)[::-1]
    labels = np.array(labels)[index_sorted]
    values = np.array(values)[index_sorted]
    indexs = np.arange(len(labels))
    
    bar_width = width
    plt.figure(figsize=size)
    plt.bar(indexs, values, color = color, lw=lw)
    plt.xticks(indexs + bar_width, labels)
    plt.title("Mots les plus fréquents au sens <<impactants>> dans 4chan", fontname="Arial", size=20, fontweight="bold")
    plt.show()
    
def wordcloud_stopwords(tokens, width, height, bg_color, size, fc_color, pad):
    stop_words = set(STOPWORDS)
    wordcloud = WordCloud(
                            width=width,
                            height= height, 
                            background_color=bg_color, 
                            stopwords=stop_words
                          ).generate(' '.join(map(str, tokens))) 

    plt.figure(figsize=size, facecolor=fc_color)
    plt.imshow(wordcloud)
    plt.axis("off") 
    plt.tight_layout(pad=pad) 
    plt.show()