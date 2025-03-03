import os
import argparse
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import torch
import re
import countries
import nltk
from itertools import chain
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer, pipeline
from nltk.corpus import stopwords
import colorsys
from itertools import combinations
import gensim
from gensim.models import Word2Vec
from transformers import pipeline, AutoTokenizer
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random
import mplcursors
from datetime import datetime
# nltk.download('punkt')
# nltk.download('stopwords')

SENTIMENT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
# "distilbert-base-uncased-finetuned-sst-2-english"

# Define a fixed color mapping for sentiment labels
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

EMBEDDING_MODEL_PATH = "word2vec_xinhua.model"

KEYWORDS = [
    # Governance & Political System
    "democracy with Chinese characteristics",
    "whole-process people's democracy",
    "socialist rule of law",
    "Western democracy",
    "authoritarianism",
    "human rights",
    "national sovereignty",
    "political stability",
    "common prosperity",
    "shared future for mankind",

    # Development & Infrastructure Initiatives
    "Belt and Road Initiative",
    "China-Pakistan Economic Corridor",
    "high-quality development",
    "infrastructure investment",
    "debt-trap diplomacy",
    "Global Development Initiative",
    "Global Security Initiative",
    "poverty alleviation",
    "sustainable development",

    # Economic Policies & Trade
    "supply-side structural reform",
    "dual circulation strategy",
    "economic decoupling",
    "Made in China 2025",
    "foreign direct investment",
    "economic globalization",
    "trade war",
    "tariffs",
    "free trade agreements",
    "export-oriented growth",

    # Technology & Innovation
    "self-reliance in technology",
    "technological sovereignty",
    "artificial intelligence",
    "5G development",
    "semiconductor industry",
    "digital economy",
    "cybersecurity",
    "data privacy",
    "smart cities",
    "China Standards 2035",

    # Soft Power & Global Influence
    "win-win cooperation",
    "multilateralism",
    "China’s peaceful rise",
    "hegemonism",
    "Western media bias",
    "geopolitical competition",
    "China-Africa relations",
    "Global South",
    "soft power diplomacy",

    # Social & Cultural Narratives
    "cultural confidence",
    "China's national rejuvenation",
    "ideological struggle",
    "universal values",
    "social harmony",
    "Chinese modernization",
    "Western decline",
    "youth patriotism"
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze sentiment and topics by region.")
    parser.add_argument("--region", type=str, required=True, 
                        choices=["europe", "africa", "asia_pacific", "north_america", "special_reports", "in_depth"],
                        help="Region to analyze (europe, africa, asia_pacific, north_america).")
    return parser.parse_args()

def preprocess_articles_2(articles):
    """Replace multi-word phrases with underscored versions and tokenize text."""
    keywords = [kw.lower() for kw in KEYWORDS]  # Lowercase keywords only here
    phrase_map = {kw: kw.replace(" ", "_") for kw in KEYWORDS}

    processed_articles = []
    for article in articles:
        article_lower = article.lower()  # Convert article text to lowercase here
        for phrase, underscored in phrase_map.items():
            article_lower = re.sub(rf"\b{re.escape(phrase)}\b", underscored, article_lower)  # Whole-word match
        processed_articles.append(article_lower.split())  # Tokenize
    return processed_articles, phrase_map

def train_word2vec(articles):
    """Train a Word2Vec model with concept-aware tokenization."""
    tokenized_articles, phrase_map = preprocess_articles_2(articles)

    w2v_model = Word2Vec(
        sentences=tokenized_articles,
        vector_size=100, window=5, min_count=2, workers=4, sg=1
    )
    return w2v_model, phrase_map

def load_models(folder_path):
    sentiment_model = pipeline(
        "text-classification", 
        model=SENTIMENT_MODEL_NAME,
        device = 0 if torch.cuda.is_available() else -1,
        top_k = None
    )
    nlp = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)

    print("Training new Word2Vec model...")
    articles = get_articles(folder_path)  # Update path
    w2v_model, phrase_map = train_word2vec(articles)
    w2v_model.save(EMBEDDING_MODEL_PATH)

    return sentiment_model, nlp, tokenizer, w2v_model, phrase_map

def get_country_list(region):
    country_dict = getattr(countries, f"countries_{region}", None)
    if country_dict is None:
        raise ValueError(f"No country list found for region: {region}")
    return country_dict

def get_articles(folder_path):
    articles = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                articles.append(text)
    
    return articles

def get_country_sentiment(sentiment_model, nlp, articles, countries):
    country_sentiment = {
        country: {"total": 0, "scores": Counter()} for country in countries
    }
    co_occurrence = Counter()

    for text in articles:
        doc = nlp(text)
        
        for paragraph in text.split("\n"):
            mentioned_countries = [c for c, keywords in countries.items() if any(k in paragraph for k in keywords)]
            for i in range(len(mentioned_countries)):
                for j in range(i + 1, len(mentioned_countries)):
                    co_occurrence[(mentioned_countries[i], mentioned_countries[j])] += 1
        
        for sentence in doc.sents:
            for country, keywords in countries.items():
                if any(k in sentence.text for k in keywords):
                    result = sentiment_model(sentence.text)

                    if result and isinstance(result, list) and isinstance(result[0], list):
                        result = result[0]  # Extract the inner list

                    top_emotion = max(result, key=lambda x: x["score"])  # Now it should work
                    country_sentiment[country]["scores"][top_emotion["label"]] += 1
                    country_sentiment[country]["total"] += 1

    return country_sentiment

def display_sentiment_results(country_sentiment):
    print("Sentiment Analysis by Country:")
    
    countries = list(country_sentiment.keys())
    top_sentiments = {}
    
    for country, data in country_sentiment.items():
        total = data["total"]
        if total > 0:
            sorted_emotions = sorted(data["scores"].items(), key=lambda x: x[1], reverse=True)
            top_sentiments[country] = sorted_emotions[:3]
            top_emotions = ", ".join([f"{k} ({v})" for k, v in sorted_emotions[:3]])
            print(f"{country}: {total} mentions | Top emotions: {top_emotions}")
    
    # Unique sentiment labels across all countries
    all_sentiments = set()
    for sentiments in top_sentiments.values():
        for sentiment, _ in sentiments:
            all_sentiments.add(sentiment)
    all_sentiments = sorted(all_sentiments)

    # Generate a color palette for the sentiments
    color_palette = sns.color_palette("tab10", len(all_sentiments))
    sentiment_colors = {sentiment: color_palette[i] for i, sentiment in enumerate(all_sentiments)}
    
    # Prepare data for stacked bar plot
    country_labels = sorted(top_sentiments.keys())  # Ensure consistent order
    sentiment_values = {sentiment: [0] * len(country_labels) for sentiment in all_sentiments}
    
    for country_idx, country in enumerate(country_labels):
        for sentiment, count in top_sentiments.get(country, []):
            sentiment_values[sentiment][country_idx] = count  # Fill the correct index

    # Stacked Bar Chart
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(country_labels))

    for sentiment in all_sentiments:
        values = sentiment_values[sentiment]
        ax.bar(country_labels, values, label=sentiment, color=sentiment_colors[sentiment], bottom=bottom)
        bottom += np.array(values)  # Stack the bars

    ax.set_xlabel("Countries")
    ax.set_ylabel("Sentiment Mentions")
    ax.set_title("Sentiment Analysis by Country (Top 3 Sentiments per Country)")
    ax.legend(title="Sentiments", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sentiment_histogram(country_sentiment):
    top_countries = sorted(country_sentiment.items(), key=lambda x: x[1]["total"], reverse=True)[:15]
    countries_list = [item[0] for item in top_countries]
    emotions_list = [max(item[1]["scores"].items(), key=lambda x: x[1], default=("None", 0))[0] for item in top_countries]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(countries_list))
    plt.bar(x, [item[1]["total"] for item in top_countries], color='blue')
    for i, (total, emotion) in enumerate(zip([item[1]["total"] for item in top_countries], emotions_list)):
        plt.text(i, total + 1, f"{countries_list[i]}\n{emotion} ({total})", ha='center', fontsize=9)
    
    plt.xlabel("Countries")
    plt.ylabel("Count of Mentions")
    plt.title("Top 15 Countries by Most Frequent Emotion")
    plt.xticks(x, countries_list, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def preprocess_articles(articles, countries):
    """ Remove country names and demonyms from articles, preserving punctuation. """
    country_words = set(word.lower() for country in countries.values() for word in country)
    
    def clean_article(article):
        return " ".join(
            word for word in re.findall(r"\b\w+\b", article) if word.lower() not in country_words
        )
    
    return [clean_article(article) for article in articles]

def topic_modeling(articles, countries={}):
    articles = preprocess_articles(articles, countries)  # Remove country names
    vectorizer = CountVectorizer(stop_words="english", max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(articles)
    terms = vectorizer.get_feature_names_out()
    
    def calculate_coherence(lda_model):
        topics = lda_model.components_
        coherence = sum(pairwise_distances(X[:, topic.argsort()[-10:]].toarray(), metric="cosine").mean()
                        for topic in topics)
        return -coherence
    
    best_n_topics, best_coherence = 2, float("-inf")
    for n in tqdm(range(2, 11)):
        lda = LatentDirichletAllocation(n_components=n, random_state=42).fit(X)
        coherence_score = calculate_coherence(lda)
        if coherence_score > best_coherence:
            best_coherence, best_n_topics = coherence_score, n
    
    lda = LatentDirichletAllocation(n_components=best_n_topics, random_state=42).fit(X)
    topics = [[terms[i] for i in topic.argsort()[-10:] if terms[i] not in countries] for topic in lda.components_]

    # Get topic distributions for each article
    topic_distributions = lda.transform(X)

    # Count number of articles assigned to each topic
    topic_counts = np.bincount(np.argmax(topic_distributions, axis=1), minlength=len(topics))

    # Plot article counts instead of word counts
    plt.figure(figsize=(10, 6))
    plt.bar([f"Topic {i}" for i in range(len(topics))], topic_counts, color="lightcoral")
    plt.xlabel("Topics")
    plt.ylabel("Number of Articles")
    plt.title("Number of Articles per Topic")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return topics, lda, X

def extract_sentiment_label(sentiment_output, exclude_neutral=True):
    """
    Extracts the correct sentiment label:
    - If exclude_neutral=True and 'neutral' is the highest, return the second highest label.
    - Otherwise, return the highest scoring label.
    """
    if not sentiment_output or not isinstance(sentiment_output, list) or not isinstance(sentiment_output[0], list):
        return "Unknown"  # Handle unexpected output format

    emotions = sorted(sentiment_output[0], key=lambda x: x['score'], reverse=True)  # Sort by score

    # If exclude_neutral is True and the top label is 'neutral', return the second-highest label
    if exclude_neutral and emotions[0]['label'] == 'neutral' and len(emotions) > 1:
        return emotions[1]['label']
    
    return emotions[0]['label']  # Otherwise, return the highest label

def analyze_sentiment_chunked(article, sentiment_model, tokenizer, exclude_neutral=True):
    """
    Splits long articles into 512-token chunks and performs sentiment analysis on each.
    Returns the most common sentiment label across chunks.
    """
    inputs = tokenizer(article, truncation=True, padding=True, max_length=512, return_tensors="pt")
    input_ids = inputs["input_ids"].squeeze()  # Get tokenized input
    num_chunks = (len(input_ids) - 1) // 512 + 1  # Calculate number of chunks

    sentiment_labels = []  # Store sentiment results

    for i in range(num_chunks):
        chunk_tokens = input_ids[i * 512 : (i + 1) * 512]  # Extract chunk
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)  # Convert back to text

        sentiment_output = sentiment_model(chunk_text)  # Run sentiment analysis
        sentiment_label = extract_sentiment_label(sentiment_output, exclude_neutral)  # Process label
        sentiment_labels.append(sentiment_label)

    # Return the most common sentiment label across all chunks
    return Counter(sentiment_labels).most_common(1)[0][0]

def sentiment_analysis_by_topic(articles, topics, lda, X, sentiment_model, tokenizer, exclude_neutral=True):
    """
    Performs sentiment analysis on articles categorized by topics.
    Uses chunking for long articles to avoid max token length errors.
    Creates interactive pie charts with hover-over article previews.
    """
    topic_distributions = lda.transform(X)
    topic_assignments = np.argmax(topic_distributions, axis=1)

    topic_sentiments = {i: [] for i in range(len(topics))}
    topic_articles = {i: [] for i in range(len(topics))}

    for i, article in enumerate(articles):
        topic_idx = topic_assignments[i]

        # Store articles for hover functionality
        topic_articles[topic_idx].append(article)

        # Handle long articles using chunking
        sentiment_label = analyze_sentiment_chunked(article, sentiment_model, tokenizer, exclude_neutral)
        topic_sentiments[topic_idx].append(sentiment_label)

    # Generate interactive pie charts for each topic
    figs = []
    for topic_idx in topic_sentiments.keys():
        sentiments = topic_sentiments[topic_idx]
        sentiment_counts = Counter(sentiments)

        if not sentiment_counts:
            labels, sizes = ["No Data"], [1]
        else:
            labels, sizes = zip(*sentiment_counts.items())

        # Select an article preview for each sentiment label
        sentiment_examples = {
            sent: random.choice([art for art, sent_val in zip(topic_articles[topic_idx], sentiments) if sent_val == sent])[:300]
            for sent in sentiment_counts.keys()
        }

        # Create DataFrame for Plotly
        df = pd.DataFrame({
            "Sentiment": labels,
            "Count": sizes,
            "Example Sentence": [sentiment_examples.get(label, "No example available") for label in labels]
        })

        # Create interactive pie chart
        fig = px.pie(df, values="Count", names="Sentiment",
                     title=f"Sentiment Distribution for Topic {topic_idx} ({', '.join(topics[topic_idx][:6])})",
                     hover_data={"Example Sentence": True},
                     labels={"Example Sentence": "Example Preview"},
                     hole=0.3)

        fig.update_traces(textinfo="percent",
                          hovertemplate="<b>%{label}</b><br>Count: %{value}<br><br>Example: %{customdata[0]}")

        figs.append(fig)

    for fig in figs:
        fig.show()

    return topic_sentiments

def associate_articles_topics(countries, articles, lda, X, topics):
    print("\nBuilding Country-Topic Matrix...")
    countries_list = list(countries.keys())
    country_topic_matrix = []
    topic_distributions = lda.transform(X)

    for topic_idx in range(len(topics)):
        print(f"Processing Topic {topic_idx}: {', '.join(topics[topic_idx])}")
        country_associations = {country: 0 for country in countries_list}

        for article_idx, article_topic_probs in enumerate(topic_distributions):
            assigned_topic = np.argmax(article_topic_probs)
            if assigned_topic == topic_idx:
                article_text = articles[article_idx]
                for country, keywords in countries.items():
                    if any(keyword.lower() in article_text.lower() for keyword in keywords):
                        country_associations[country] += 1

        country_topic_matrix.append([country_associations[country] for country in countries_list])

    return country_topic_matrix

def visualize_articles_topics(country_topic_matrix, countries, topics):
    countries_list = list(countries.keys())
    country_topic_array = np.array(country_topic_matrix)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        country_topic_array.T, 
        annot=True,
        fmt="d",
        cmap="coolwarm",
        xticklabels=[f"Topic {i}" for i in range(len(topics))],
        yticklabels=countries_list,
        cbar_kws={"label": "Number of Articles"}
    )
    plt.title("Country-Topic Associations Heatmap")
    plt.xlabel("Topics")
    plt.ylabel("Countries")
    plt.tight_layout()
    plt.show()

def get_country_cooccurrence(countries, articles, nlp):
    # Initialize a co-occurrence matrix
    co_occurrence = Counter()

    countries["China"] = ["China", "Chinese", "CCP"]

    for text in articles:
        # Process the text with spaCy to extract sentences
        doc = nlp(text)
        for paragraph in text.split("\n"):
            mentioned_countries = []
            for country, keywords in countries.items():
                if any(keyword in paragraph for keyword in keywords):
                    mentioned_countries.append(country)

            # Update co-occurrence counts
            for i in range(len(mentioned_countries)):
                for j in range(i + 1, len(mentioned_countries)):
                    co_occurrence[(mentioned_countries[i], mentioned_countries[j])] += 1

    return co_occurrence

def visualize_country_cooccurrence(co_occurrence):
    # Visualize country co-occurrence as a graph
    print("\nVisualizing Country Co-occurrence...")
    G = nx.Graph()

    for (country1, country2), count in co_occurrence.items():
        if count > 0:  # Only include co-occurrences with at least one mention
            G.add_edge(country1, country2, weight=count)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))

    # Draw nodes and edges with edge thickness representing weight
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # Add edge labels for weights
    edge_labels = {(u, v): f"{data['weight']}" for u, v, data in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Country Co-occurrence Graph")
    plt.show()

def cross_continent_sentiment_analysis(sentiment_model, nlp, base_folder):
    """Compare sentiment trends across different continents."""
    regions = ["europe", "africa", "asia_pacific", "north_america"]
    continent_sentiments = {}

    for region in regions:
        print(f"\nProcessing {region}...")
        articles = get_articles(os.path.join(base_folder, f"test_articles_{region}"))
        countries = get_country_list(region)
        country_sentiments = get_country_sentiment(sentiment_model, nlp, articles, countries)
        
        # Aggregate sentiment scores for the region
        region_sentiment = Counter()
        total_mentions = 0

        for country_data in country_sentiments.values():
            region_sentiment.update(country_data["scores"])
            total_mentions += country_data["total"]

        continent_sentiments[region] = {
            "total_mentions": total_mentions,
            "sentiment_scores": region_sentiment
        }

    # Visualization
    plt.figure(figsize=(10, 6))
    for region, data in continent_sentiments.items():
        sentiments, counts = zip(*data["sentiment_scores"].most_common(5))  # Top 5 sentiments
        plt.bar(sentiments, counts, label=region, alpha=0.7)

    plt.xlabel("Sentiments")
    plt.ylabel("Mentions")
    plt.title("Cross-Continent Sentiment Comparison")
    plt.legend()
    plt.show()

def truncate_text(text, tokenizer, max_length=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def filter_meaningful_words(word_list):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in word_list if word.isalnum() and word.lower() not in stop_words]
    return filtered_words

def generate_rgb_colors(num_colors):
    colors = []
    for i in range(num_colors):
        # Generate evenly spaced hues in the HSV color space
        hue = float(i) / float(num_colors)  # hue values range from 0 to 1
        saturation = 0.8  # moderate saturation to avoid overly gray colors
        value = 0.9  # high value for bright colors
        
        # Convert to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Ensure that colors are correctly scaled (shouldn't be clipped to red)
        colors.append(rgb)
    return colors

def analyze_us_china_mentions(articles, sentiment_model, tokenizer, nlp):
    us_terms = {"United States", "USA", "America", "U.S.", "Washington", "Trump", "Biden", 
            "The States", "American government", "White House", "Congress", "Democrats", 
            "Republicans", "U.S. Congress", "U.S. Senate", "U.S. House of Representatives", 
            "FBI", "CIA"}
    china_terms = {"China", "PRC", "Beijing", "Xi Jinping", "CCP", "Chinese government", 
                "People's Republic of China", "Great Wall", "Shanghai", "Hong Kong", "Macau", 
                "Chinese Communist Party", "Chinese President", "CPC", "PLA", "Confucius Institute"}

    us_context = []
    china_context = []
    us_sentiments = Counter()
    china_sentiments = Counter()
    
    for article in articles:
        sentences = sent_tokenize(article)
        
        for sent in sentences:
            words = set(word_tokenize(sent))
            truncated_sent = truncate_text(sent, tokenizer)  # Ensure proper truncation
            sentiment_score = sentiment_model(truncated_sent)[0]
            meaningful_words = filter_meaningful_words(words)

            if sentiment_score:
                # Sort the sentiment scores in descending order based on the 'score' value
                sorted_scores = sorted(sentiment_score, key=lambda x: x['score'], reverse=True)
                
                # Check if the highest sentiment is 'neutral', if so, select the second highest
                if sorted_scores[0]['label'] == "neutral" and len(sorted_scores) > 1:
                    top_sentiment = sorted_scores[1]['label']
                else:
                    top_sentiment = sorted_scores[0]['label']
            else:
                top_sentiment = "neutral"
            
            if words & us_terms:
                us_context.extend(meaningful_words)
                us_sentiments[top_sentiment] += 1
            if words & china_terms:
                china_context.extend(meaningful_words)
                china_sentiments[top_sentiment] += 1
    
    # Extract most common co-occurring words
    us_common_words = Counter(us_context).most_common(20)
    china_common_words = Counter(china_context).most_common(20)

    # Define the sentiment categories and their values
    labels = list(set(us_sentiments.keys()) | set(china_sentiments.keys()))
    # Create a color palette using seaborn or matplotlib (depending on the number of sentiments)
    num_sentiments = len(labels)
    colors = generate_rgb_colors(num_sentiments)
    # Values for each sentiment, defaulting to 0 if the sentiment doesn't appear in a dictionary
    us_values = [us_sentiments.get(label, 0) for label in labels]
    china_values = [china_sentiments.get(label, 0) for label in labels]

    # Create a figure with subplots for each pie chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # U.S. Sentiment Pie Chart
    axes[0].pie(us_values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title("U.S. Sentiment Distribution")

    # China Sentiment Pie Chart
    axes[1].pie(china_values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title("China Sentiment Distribution")

    # Display the pie charts
    plt.tight_layout()
    plt.show()
    
    return {
        "us_common_words": us_common_words,
        "china_common_words": china_common_words,
        "us_sentiment_summary": us_sentiments,
        "china_sentiment_summary": china_sentiments
    }

def extract_relevant_sentences(articles, keywords):
    """
    Extract sentences from articles that contain any of the specified keywords.
    """
    relevant_sentences = []
    pattern = re.compile(r'\b(' + '|'.join([re.escape(kw) for kw in keywords]) + r')\b', re.IGNORECASE)
    
    for article in articles:
        sentences = sent_tokenize(article)  # Split article into sentences
        for sentence in sentences:
            if pattern.search(sentence):  # If keyword is found, add sentence
                relevant_sentences.append(sentence)
    
    return relevant_sentences

def analyze_sentiment(sentences, sentiment_model):
    """
    Perform sentiment analysis on a list of sentences using Hugging Face pipeline.
    """
    sentiments = []
    for sentence in sentences:
        result = sentiment_model(sentence)
        sentiments.append(extract_sentiment_label(result))  # Extract sentiment label
    
    return sentiments

def plot_sentiment_keywords(sentiments, sentences, topic_name):
    """
    Plot an interactive pie chart showing sentiment distribution with tooltips
    displaying example sentences when hovering over slices.
    """
    # Count sentiment labels
    sentiment_counts = Counter(sentiments)
    
    # Extract labels and sizes
    labels, sizes = zip(*sentiment_counts.items()) if sentiment_counts else (["No Data"], [1])

    # Select an example sentence for each sentiment
    sentiment_examples = {sent: random.choice([s for s, sent_val in zip(sentences, sentiments) if sent_val == sent])
                          for sent in sentiment_counts.keys()}

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        "Sentiment": labels,
        "Count": sizes,
        "Example Sentence": [sentiment_examples.get(label, "") for label in labels]    
        })

    # Create interactive pie chart
    fig = px.pie(df, values="Count", names="Sentiment", title=f"Sentiment of sentences related to {topic_name}",
                 hover_data={"Example Sentence": True}, 
                 labels={"Example Sentence": "Example Sentence"},
                 hole=0.3)  # Optional: creates a donut chart

    fig.update_traces(textinfo="percent", hovertemplate="<b>%{label}</b><br>Count: %{value}<br><br>Example: %{customdata[0]}")

    fig.show()

def sentiment_analysis_keywords(articles, sentiment_model, keywords, topic_name):
    """
    Extract relevant sentences mentioning BRI or Western-led initiatives, 
    analyze sentiment, and compare results.
    """
    
    # Extract relevant sentences mentioning these keywords
    sentences = extract_relevant_sentences(articles, keywords)

    print(sentences[:10])
    
    # Perform sentiment analysis on extracted sentences
    sentiments = analyze_sentiment(sentences, sentiment_model)
    
    # Plot results
    plot_sentiment_keywords(sentiments, sentences, topic_name)
    
    # Return results
    return {
        "Sentiments": Counter(sentiments)
    }

def get_similarity(keyword, countries, w2v_model):
    """Compute cosine similarity between a keyword and each country."""
    keyword = keyword.replace(" ", "_").lower()

    if keyword not in w2v_model.wv:
        return {}

    keyword_vec = w2v_model.wv[keyword]
    similarities = {}

    for country, variations in countries.items():  # Use .items() for dict iteration
        country_lower = country.replace(" ", "_").lower()
        best_sim = None

        # Check if main country name is in vocabulary
        if country_lower in w2v_model.wv:
            country_vec = w2v_model.wv[country_lower]
            best_sim = np.dot(keyword_vec, country_vec) / (np.linalg.norm(keyword_vec) * np.linalg.norm(country_vec))

        # If main name is missing, check variations
        for variation in variations:
            variation_lower = variation.replace(" ", "_").lower()
            if variation_lower in w2v_model.wv:
                variation_vec = w2v_model.wv[variation_lower]
                cosine_sim = np.dot(keyword_vec, variation_vec) / (np.linalg.norm(keyword_vec) * np.linalg.norm(variation_vec))

                # Keep the highest similarity
                if best_sim is None or cosine_sim > best_sim:
                    best_sim = cosine_sim

        similarities[country] = best_sim  # Store best similarity found

    return similarities

def get_all_similarities(countries, w2v_model):
    """Computes similarity scores for each keyword-country pair and stores in a dictionary."""
    print("Computing similarities")
    similarities = {}

    for keyword in KEYWORDS:
        similarities[keyword] = get_similarity(keyword, countries, w2v_model)

    return similarities

def get_sample_sentence(keyword, country, articles):
    """
    Find a sample sentence from the dataset that contains both the keyword and country.

    Args:
        keyword (str): The keyword to search for.
        country (str): The country name to search for.
        articles (list): List of article texts.

    Returns:
        str: A sample sentence containing the keyword and country.
    """

    keyword = keyword.lower()
    country = country.lower()

    relevant_sentences = []
    partial_sentences = []

    for article in articles:
        sentences = sent_tokenize(article)  # Split article into sentences

        for sentence in sentences:
            sentence_lower = sentence.lower()

            if keyword in sentence_lower and country in sentence_lower:
                relevant_sentences.append(sentence)  # Best match
            elif keyword in sentence_lower or country in sentence_lower:
                partial_sentences.append(sentence)  # Partial match

    # Return the best matching sentence
    if relevant_sentences:
        return random.choice(relevant_sentences)  # Pick a random relevant sentence
    elif partial_sentences:
        return random.choice(partial_sentences)  # Pick a partial match
    else:
        return f"No relevant sentence found for '{keyword}' and '{country}'."

def visualize_similarities(similarities, articles):
    """Generates an interactive heatmap with hover-over sample sentences."""

    print("Visualizing heatmap...")
    
    # Convert dictionary to DataFrame and replace None with NaN
    df = pd.DataFrame(similarities).T
    df = df.map(lambda x: np.nan if x is None else x)
    
    # Drop rows & columns that contain only NaN values
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # If no valid data remains, return early
    if df.empty:
        print("No valid similarities to display.")
        return
    
    # Create a hover text matrix with sample sentences
    hover_text = df.apply(lambda row: [get_sample_sentence(row.name, col, articles) for col in df.columns], axis=1)
    
    # Create interactive heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=df.values,  # Similarity scores
        x=df.columns,  # Country names
        y=df.index,  # Keywords
        colorscale="RdBu",
        text=hover_text,  # Sample sentences
        hoverinfo="text"  # Display hover text
    ))

    # Update layout for better readability
    fig.update_layout(
        title="Keyword-Country Association Heatmap",
        xaxis_title="Countries",
        yaxis_title="Keywords",
        xaxis=dict(tickangle=-45),
        autosize=True
    )

    # Show interactive heatmap
    fig.show()

def get_articles_by_date(folder_path):
    articles_2023 = []
    pre_nov_2024 = []
    post_nov_2024 = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            try:
                # Extract the date part from the filename
                date_str = filename.split("_")[0]  # Get 'YYYY-MM-DD HH:MM:SS'
                article_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

                # Read article content
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                
                # Categorize based on date
                if article_date.year == 2023:
                    articles_2023.append(text)
                elif article_date < datetime(2024, 11, 1):
                    pre_nov_2024.append(text)
                else:
                    post_nov_2024.append(text)
                    
            except ValueError:
                print(f"Skipping file due to incorrect date format: {filename}")

    return articles_2023, pre_nov_2024, post_nov_2024

def main():
    args = parse_arguments()
    region = args.region

    folder_path = f"./articles_{region}"
    countries = get_country_list('generic') # TODO: add a conditional
    sentiment_model, nlp, tokenizer, w2v_model, phrase_map = load_models(folder_path)
    
    articles = get_articles(folder_path)

    """
    # Sentiment Analysis
    country_sentiment = get_country_sentiment(sentiment_model, nlp, articles, countries)  
    display_sentiment_results(country_sentiment)
    plot_sentiment_histogram(country_sentiment)
    """

    # Topic Modeling
    topics, lda, X = topic_modeling(articles)
    print("\nTopics discovered:")
    for idx, topic in enumerate(topics):
        print(f"Topic {idx}: {', '.join(topic)}")

    """
    topic_sentiments = sentiment_analysis_by_topic(articles, topics, lda, X, sentiment_model, tokenizer, exclude_neutral=False)
    topic_sentiments = sentiment_analysis_by_topic(articles, topics, lda, X, sentiment_model, tokenizer)
    """

    # Articles by topic
    country_topic_matrix = associate_articles_topics(countries, articles, lda, X, topics)
    visualize_articles_topics(country_topic_matrix, countries, topics)

    # Country co-occurrence
    co_occurrence = get_country_cooccurrence(countries, articles, nlp)
    visualize_country_cooccurrence(co_occurrence)

    # U.S. vs. China mentions
    us_china_mentions = analyze_us_china_mentions(articles, sentiment_model, tokenizer, nlp)
    print(us_china_mentions)

    # Define keyword lists
    western_keywords = [
    "USAID", "World Bank", "Western aid", "Western-led", "IMF", "OECD", 
    "International Monetary Fund", "United States assistance", "G7 infrastructure plan",
    "Build Back Better World", "B3W", "Western-backed loans", "Western development aid",
    "Western financing", "World Bank loans", "G20 development projects", "Western investment",
    "Washington Consensus", "EU development assistance", "European aid programs",
    "US foreign assistance", "United States-funded projects", "NATO development aid"
    ]
    bri_keywords = [
    "Belt and Road", "BRI", "Silk Road", "China-Africa cooperation",
    "Chinese development", "Chinese initiative", "Asian Infrastructure Investment Bank",
    "AIIB", "China-led development", "Beijing Consensus", "China-Africa partnership",
    "China-backed", "China-financed", "China’s investment", "China-funded", 
    "China’s economic strategy", "China’s infrastructure projects", 
    "China’s global strategy", "China’s overseas investment"
    ]


    western_results = sentiment_analysis_keywords(articles, sentiment_model, western_keywords, "Western Development Initiatives")
    bri_results = sentiment_analysis_keywords(articles, sentiment_model, bri_keywords, "Chinese Development Initiatives")

    similarities = get_all_similarities(countries, w2v_model)

    # Choose one:
    visualize_similarities(similarities, articles) 
    # print out some examples of sentences these "Similar words" occur in

    

    _2023, pre_nov_2024, post_nov_2024 = get_articles_by_date(folder_path)
    sets = [_2023, pre_nov_2024, post_nov_2024]

    """
    for articles in sets:
        us_china_mentions = analyze_us_china_mentions(articles, sentiment_model, tokenizer, nlp)
        print(us_china_mentions)

        western_results = sentiment_analysis_keywords(articles, sentiment_model, western_keywords, "Western Development Initiatives")
        bri_results = sentiment_analysis_keywords(articles, sentiment_model, bri_keywords, "Chinese Development Initiatives")

        similarities = get_all_similarities(countries, w2v_model)

        # Choose one:
        visualize_similarities(similarities, articles) 
        # print out some examples of sentences these "Similar words" occur in
    """


if __name__ == "__main__":
    main()