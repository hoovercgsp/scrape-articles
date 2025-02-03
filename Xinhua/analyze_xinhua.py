import os
import argparse
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import torch
import re
import countries

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze sentiment and topics by region.")
    parser.add_argument("--region", type=str, required=True, 
                        choices=["europe", "africa", "asia_pacific", "north_america"],
                        help="Region to analyze (europe, africa, asia_pacific, north_america).")
    return parser.parse_args()


def load_models():
    sentiment_model = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base",  # More nuanced emotion model
        device = 0 if torch.cuda.is_available() else -1,
        top_k = None
    )
    nlp = spacy.load("en_core_web_sm")
    return sentiment_model, nlp

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

def topic_modeling(articles, countries):
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

def main():
    args = parse_arguments()
    region = args.region

    folder_path = f"./articles_{region}"
    countries = get_country_list(region)
    sentiment_model, nlp = load_models()
    
    articles = get_articles(folder_path)

    # Sentiment Analysis
    country_sentiment = get_country_sentiment(sentiment_model, nlp, articles, countries)  
    display_sentiment_results(country_sentiment)
    plot_sentiment_histogram(country_sentiment)

    # Topic Modeling
    topics, lda, X = topic_modeling(articles, countries)
    print("\nTopics discovered:")
    for idx, topic in enumerate(topics):
        print(f"Topic {idx}: {', '.join(topic)}")

    # Articles by topic
    country_topic_matrix = associate_articles_topics(countries, articles, lda, X, topics)
    visualize_articles_topics(country_topic_matrix, countries, topics)

    # Country co-occurrence
    co_occurrence = get_country_cooccurrence(countries, articles, nlp)
    visualize_country_cooccurrence(co_occurrence)

if __name__ == "__main__":
    main()
