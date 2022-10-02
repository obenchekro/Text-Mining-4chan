import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiment_score(df, n):
    sentiment_object = SentimentIntensityAnalyzer()
    sentiment_score = {index : {
                                 data["Reply"] : sentiment_object.polarity_scores(data["Reply"])
                                } for index, data in df.iterrows()} 
    return list(sentiment_score.items())[:n]

def get_polarity_df(df, n):
    df['Compound Score'] = pd.to_numeric(df['Reply'].apply(lambda data : SentimentIntensityAnalyzer().polarity_scores(data)['compound']))
    df['Polarity'] = np.where(df['Compound Score'] >= 0.05, 'Positive',(np.where(df['Compound Score'] <= -0.05, 'Negative', 'Neutral')))
    return df[:n]

def hist_polarity_content(df, color):
    return df['Polarity'].value_counts().plot(kind='bar', color=color)

def hist_polarity_category(df, size, color_pos, color_neg, color_neu):
    df_positive = df.loc[(df['Polarity'] == 'Positive')]
    df_negative = df.loc[(df['Polarity'] == 'Negative')]
    df_neutral = df.loc[(df['Polarity'] == 'Neutral')]                  
                         
    fig, axes = plt.subplots(3, figsize=size)
    fig.suptitle("Fréquence des polarités par catégorie des topics", fontname="Arial", size=20, fontweight="bold")
    
    df_positive.groupby(['Acronym Category'])['Polarity'].count().plot(kind='barh', ax=axes[0], color=color_pos)
    axes[0].set_title("Fréquence de la polarité positive par catégorie", fontname="Arial", size=15)
    df_negative.groupby(['Acronym Category'])['Polarity'].count().plot(kind='barh', ax=axes[1], color=color_neg)
    axes[1].set_title("Fréquence de la polarité négative par catégorie", fontname="Arial", size=15)
    df_neutral.groupby(['Acronym Category'])['Polarity'].count().plot(kind='barh', ax=axes[2], color=color_neu)
    axes[2].set_title("Fréquence de la polarité neutre par catégorie", fontname="Arial", size=15)

    return fig, axes

