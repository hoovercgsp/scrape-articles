import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from collections import Counter
from typing import Dict, List, Tuple
from utils import Topic, Article
import datetime

SENTIMENT_COLORS = {
    "neutral": "gray",
    "joy": "yellow",
    "anger": "red",
    "sadness": "blue",
    "fear": "purple",
    "disgust": "brown",
    "surprise": "orange",
    "Unknown": "black"  # Fallback color
}

def plot_sentiment_by_category(sentiment_data: Dict[str, Dict[str, int]], title: str):
    """
    Creates a stacked bar chart for sentiment distribution across categories.

    Parameters:
    - sentiment_data: Dict[str, Dict[str, int]]
        - Key = category (e.g., country or keyword set)
        - Value = Dictionary mapping sentiment labels to counts
    - title: str
    """
    categories = list(sentiment_data.keys())  # Ensure it's a list of category names (strings)
    sentiments = sorted(set(s for cat in sentiment_data.values() for s in cat))  # Sort for consistency

    # Convert category names explicitly to strings
    categories_str = [str(cat) for cat in categories]  

    sentiment_counts = {s: np.array([sentiment_data[cat].get(s, 0) for cat in categories]) for s in sentiments}

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(categories))

    for sentiment in sentiments:
        counts = sentiment_counts[sentiment]
        ax.bar(categories_str, counts, bottom=bottom, color=SENTIMENT_COLORS.get(sentiment, "black"), label=sentiment)
        bottom += counts

    ax.set_ylabel("Sentiment Count")
    ax.set_title(title)
    ax.legend(title="Sentiments")
    plt.xticks(rotation=45)
    plt.show()

def plot_topic_distribution(topic_data: Dict[Topic, List], title: str):
    """
    Creates a pie chart for topic distribution.

    Parameters:
    - topic_data: Dict[Tuple[str, ...], List]
        - Key = topic (Topic)
        - Value = list of articles categorized under this topic
    - title: str
    """
    topic_labels = [' '.join(topic.title) for topic in topic_data.keys()]
    topic_sizes = [len(articles) for articles in topic_data.values()]

    plt.figure(figsize=(8, 8))
    plt.pie(topic_sizes, labels=topic_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title(title)
    plt.show()

def plot_keyword_context(keyword_contexts: Dict[str, Counter], keyword: str, max_words: int = 20):
    """
    Creates a word cloud for the most common words appearing near a given keyword.

    Parameters:
    - keyword_contexts: Dict[str, Counter]
        - Key = keyword
        - Value = Counter of words appearing in its context
    - keyword: str
    - max_words: int (default 20)
    """
    if keyword not in keyword_contexts:
        print(f"No context words found for '{keyword}'")
        return
    
    word_frequencies = dict(keyword_contexts[keyword].most_common(max_words))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_frequencies)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for Context of '{keyword}'")
    plt.show()

def plot_compare_sentiments(sentiment_data: Dict[str, Dict[str, int]], title: str):
    """
    Creates a grouped bar chart to compare sentiment distribution across different keyword sets.

    Parameters:
    - sentiment_data: Dict[str, Dict[str, int]]
        - Key = keyword set (e.g., 'Economic', 'Political', etc.)
        - Value = Dictionary mapping sentiment labels to counts
    - title: str
    """
    sentiment_labels = list({s for topic in sentiment_data.values() for s in topic})
    categories = list(sentiment_data.keys())
    x = np.arange(len(sentiment_labels))  # Bar positions

    width = 0.2  # Width of bars
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, category in enumerate(categories):
        sentiment_counts = [sentiment_data[category].get(s, 0) for s in sentiment_labels]
        ax.bar(x + i * width, sentiment_counts, width, label=category, color=plt.cm.Paired(i))

    ax.set_xticks(x + width * (len(categories) / 2 - 0.5))
    ax.set_xticklabels(sentiment_labels, rotation=45)
    ax.set_ylabel("Sentiment Count")
    ax.set_title(title)
    ax.legend(title="Keyword Sets")
    plt.show()


def compare_keyword_sentiments(sentiment_data: Dict[str, Dict[str, Counter]], title: str):
    """
    Creates a grouped bar chart to compare sentiment distribution across different keyword sets.

    Parameters:
    - sentiment_data: Dict[str, Dict[str, Counter]]
        - Key = keyword group name (e.g., 'Economic', 'Political')
        - Value = Dictionary with a "Sentiments" key, mapping sentiment labels to counts.
    - title: str
    """
    # Extract keyword groups and sentiment labels
    keyword_groups = list(sentiment_data.keys())
    sentiment_labels = sorted(set(sent for group in sentiment_data.values() for sent in group["Sentiments"]))

    x = np.arange(len(sentiment_labels))  # X positions for sentiment categories
    width = 0.15  # Width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over keyword groups and plot bars
    for i, group in enumerate(keyword_groups):
        sentiment_counts = [sentiment_data[group]["Sentiments"].get(sent, 0) for sent in sentiment_labels]
        ax.bar(x + i * width, sentiment_counts, width, label=group, color=plt.cm.Paired(i))

    # Formatting the plot
    ax.set_xticks(x + width * (len(keyword_groups) / 2 - 0.5))
    ax.set_xticklabels(sentiment_labels, rotation=45)
    ax.set_ylabel("Sentiment Count")
    ax.set_title(title)
    ax.legend(title="Keyword Groups")
    plt.show()

"""
--- TIME SERIES ANALYSIS
"""


def plot_sentiment_over_time(sentiment_series: Dict[datetime, Dict[str, int]], title: str):
    """
    Plots a stacked area chart showing the distribution of sentiments over time.

    Parameters:
    - sentiment_series: Dict[datetime, Dict[str, int]]
        - Key = timestamp (datetime)
        - Value = Dictionary mapping sentiment labels to counts
    - title: str
    """
    df = pd.DataFrame(sentiment_series).T.fillna(0)
    df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    df.plot(kind="area", stacked=True, colormap="coolwarm", alpha=0.75, figsize=(12, 6))
    
    plt.xlabel("Time")
    plt.ylabel("Sentiment Count")
    plt.title(title)
    plt.legend(title="Sentiments")
    plt.xticks(rotation=45)
    plt.show()

def plot_topic_trends_over_time(topic_series: Dict[datetime, Dict[str, int]], title: str):
    """
    Plots a line chart showing the frequency of different topics over time.

    Parameters:
    - topic_series: Dict[datetime, Dict[str, int]]
        - Key = timestamp (datetime)
        - Value = Dictionary mapping topic labels to counts
    - title: str
    """
    df = pd.DataFrame(topic_series).T.fillna(0)
    df.index = pd.to_datetime(df.index)
    
    df.sort_index(inplace=True)
    df.plot(kind="line", marker="o", figsize=(12, 6), linewidth=2, colormap="tab10")
    
    plt.xlabel("Time")
    plt.ylabel("Topic Mentions")
    plt.title(title)
    plt.legend(title="Topics")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def plot_keyword_frequency_heatmap(keyword_series: Dict[datetime, Dict[str, int]], title: str):
    """
    Plots a heatmap showing keyword frequency over time.

    Parameters:
    - keyword_series: Dict[datetime, Dict[str, int]]
        - Key = timestamp (datetime)
        - Value = Dictionary mapping keywords to counts
    - title: str
    """
    df = pd.DataFrame(keyword_series).T.fillna(0)
    df.index = pd.to_datetime(df.index)

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.T, cmap="Blues", linewidths=0.5, annot=True, fmt="d")

    plt.xlabel("Time")
    plt.ylabel("Keywords")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def plot_moving_average_sentiment(sentiment_series: Dict[datetime, Dict[str, int]], window: int = 3, title: str = "Sentiment Moving Average"):
    """
    Plots a smoothed sentiment trend using a moving average.

    Parameters:
    - sentiment_series: Dict[datetime, Dict[str, int]]
        - Key = timestamp (datetime)
        - Value = Dictionary mapping sentiment labels to counts
    - window: int
        - Window size for moving average
    - title: str
    """
    df = pd.DataFrame(sentiment_series).T.fillna(0)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    smoothed_df = df.rolling(window=window, min_periods=1).mean()

    smoothed_df.plot(kind="line", figsize=(12, 6), linewidth=2, colormap="coolwarm")
    
    plt.xlabel("Time")
    plt.ylabel("Smoothed Sentiment Count")
    plt.title(title)
    plt.legend(title="Sentiments")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def plot_sentiment_correlation(sentiment_series: Dict[datetime, Dict[str, int]], title: str = "Sentiment Correlation Over Time"):
    """
    Plots a correlation heatmap for sentiment trends over time.

    Parameters:
    - sentiment_series: Dict[datetime, Dict[str, int]]
        - Key = timestamp (datetime)
        - Value = Dictionary mapping sentiment labels to counts
    - title: str
    """
    df = pd.DataFrame(sentiment_series).T.fillna(0)
    df.index = pd.to_datetime(df.index)

    correlation_matrix = df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)

    plt.title(title)
    plt.show()