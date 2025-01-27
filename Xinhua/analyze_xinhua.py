import os
from transformers import pipeline
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm  # For progress bar visualization
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Load sentiment analysis model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Folder containing the articles
folder_path = "./test_articles_europe"

# List of European countries and their derivatives
countries = {
    "Albania": ["Albania", "Albanian"],
    "Andorra": ["Andorra", "Andorran"],
    "Austria": ["Austria", "Austrian"],
    "Belarus": ["Belarus", "Belarusian"],
    "Belgium": ["Belgium", "Belgian"],
    "Bosnia and Herzegovina": ["Bosnia", "Bosnian", "Herzegovina", "Herzegovinian"],
    "China" : ["China", "Chinese", "CCP"],
    "Bulgaria": ["Bulgaria", "Bulgarian"],
    "Croatia": ["Croatia", "Croatian"],
    "Cyprus": ["Cyprus", "Cypriot"],
    "Czech Republic": ["Czech Republic", "Czechia", "Czech"],
    "Denmark": ["Denmark", "Danish", "Dane"],
    "Estonia": ["Estonia", "Estonian"],
    "Finland": ["Finland", "Finnish", "Finn"],
    "France": ["France", "French"],
    "Germany": ["Germany", "German"],
    "Greece": ["Greece", "Greek"],
    "Hungary": ["Hungary", "Hungarian"],
    "Iceland": ["Iceland", "Icelandic"],
    "Ireland": ["Ireland", "Irish"],
    "Italy": ["Italy", "Italian"],
    "Latvia": ["Latvia", "Latvian"],
    "Liechtenstein": ["Liechtenstein", "Liechtensteiner"],
    "Lithuania": ["Lithuania", "Lithuanian"],
    "Luxembourg": ["Luxembourg", "Luxembourgish"],
    "Malta": ["Malta", "Maltese"],
    "Moldova": ["Moldova", "Moldovan"],
    "Monaco": ["Monaco", "Monégasque", "Monegasque"],
    "Montenegro": ["Montenegro", "Montenegrin"],
    "Netherlands": ["Netherlands", "Dutch"],
    "North Macedonia": ["North Macedonia", "Macedonia", "Macedonian"],
    "Norway": ["Norway", "Norwegian"],
    "Poland": ["Poland", "Polish"],
    "Portugal": ["Portugal", "Portuguese"],
    "Romania": ["Romania", "Romanian"],
    "Russia": ["Russia", "Russian"],
    "San Marino": ["San Marino", "Sammarinese"],
    "Serbia": ["Serbia", "Serbian"],
    "Slovakia": ["Slovakia", "Slovak"],
    "Slovenia": ["Slovenia", "Slovenian"],
    "Spain": ["Spain", "Spanish"],
    "Sweden": ["Sweden", "Swedish", "Swede"],
    "Switzerland": ["Switzerland", "Swiss"],
    "Turkey": ["Turkey", "Turkish"],
    "Ukraine": ["Ukraine", "Ukrainian"],
    "United Kingdom": ["United Kingdom", "UK", "British", "England", "Scotland", "Wales", "Northern Ireland", "English", "Scottish", "Welsh", "Irish"],
    "Vatican City": ["Vatican City", "Vatican", "Holy See"]
}

# Initialize a dictionary to store sentiment scores and counts by country
country_sentiment = {country: {"total": 0, "POSITIVE": 0, "NEGATIVE": 0} for country in countries}

# Collect all articles for topic modeling
articles = []

# Initialize a co-occurrence matrix
co_occurrence = Counter()

# Loop through all .txt files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        # Read the article content
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            articles.append(text)

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

        for sentence in doc.sents:
            sentence_text = sentence.text

            # Check if any country or its derivatives are mentioned in the sentence
            for country, keywords in countries.items():
                if any(keyword in sentence_text for keyword in keywords):
                    # Perform sentiment analysis on the sentence
                    result = sentiment_model(sentence_text)
                    label = result[0]["label"]

                    # Update sentiment scores and counts for the country
                    if label == "POSITIVE":
                        country_sentiment[country]["POSITIVE"] += 1
                    else:
                        country_sentiment[country]["NEGATIVE"] += 1
                    country_sentiment[country]["total"] += 1

# Calculate sentiment percentages and display the results
print("Sentiment Analysis by Country:")
for country, scores in country_sentiment.items():
    total = scores.get("total", 0)
    if total > 0:
        positive_pct = (scores.get("POSITIVE", 0) / total) * 100
        print(f"{country}: {total} mentions ({positive_pct:.2f}% positive)")
    else:
        print(f"{country}: No mentions")

# Filter only countries with mentions and calculate sentiment counts
sentiment_data = {
    country: (
        scores.get("total", 0), 
        scores.get("POSITIVE", 0), 
        scores.get("NEGATIVE", 0),
        (scores.get("POSITIVE", 0) / scores.get("total", 1)) * 100  # Positive percentage
    )
    for country, scores in country_sentiment.items()
    if scores.get("total", 0) > 0
}

# Sort by total mentions and get the top 15 countries
sorted_data = sorted(sentiment_data.items(), key=lambda x: x[1][0], reverse=True)[:15]

# Unpack filtered data
countries_list, total_mentions, positive_counts, negative_counts, positive_pcts = zip(*[
    (country, data[0], data[1], data[2], data[3]) for country, data in sorted_data
])

# Plot the histogram
x = np.arange(len(countries_list))  # the label locations
width = 0.8  # bar width

plt.figure(figsize=(12, 8))
plt.bar(x, positive_counts, width, label='Positive Mentions', color='green')
plt.bar(x, negative_counts, width, bottom=positive_counts, label='Negative Mentions', color='red')

# Add country names with positive percentage above each bar
for i, (total, pct) in enumerate(zip(total_mentions, positive_pcts)):
    plt.text(i, total + 1, f"{countries_list[i]}\n({total}, {pct:.1f}%)", ha='center', fontsize=9)

plt.xlabel("Countries")
plt.ylabel("Count of Mentions")
plt.title("Top 15 Countries by Sentiment Mentions")
plt.xticks(x, countries_list, rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

# Topic Modeling with LDA
print("\nPerforming Topic Modeling...")
# Preprocessing articles
vectorizer = CountVectorizer(stop_words="english", max_df=0.95, min_df=2)
X = vectorizer.fit_transform(articles)

# Function to calculate the coherence score for LDA
def calculate_coherence_score(lda_model, vectorized_data, terms, top_n_words=10):
    topics = lda_model.components_
    coherence = 0
    for topic_idx, topic in enumerate(topics):
        top_words_idx = topic.argsort()[-top_n_words:]
        top_words = [terms[i] for i in top_words_idx]
        # Calculate pairwise distances for the top words
        distances = pairwise_distances(
            vectorized_data[:, top_words_idx].toarray(), metric="cosine"
        )
        coherence += distances.mean()
    return -coherence  # Return as negative since lower is better

# Determine the optimal number of topics
print("\nFinding the optimal number of topics...")
min_topics = 2
max_topics = 10
best_n_topics = min_topics
best_coherence = float("-inf")

terms = vectorizer.get_feature_names_out()

for n_topics in tqdm(range(min_topics, max_topics + 1)):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    coherence_score = calculate_coherence_score(lda, X, terms)
    print(f"Number of Topics: {n_topics}, Coherence Score: {coherence_score:.4f}")
    if coherence_score > best_coherence:
        best_coherence = coherence_score
        best_n_topics = n_topics

print(f"\nOptimal number of topics: {best_n_topics} (Coherence Score: {best_coherence:.4f})")

# Run LDA with the optimal number of topics
lda = LatentDirichletAllocation(n_components=best_n_topics, random_state=42)
lda.fit(X)

# Display topics
print("\nTopics discovered:")
topics = []
for idx, topic in enumerate(lda.components_):
    top_terms = [terms[i] for i in topic.argsort()[-10:]]
    topics.append(top_terms)
    print(f"Topic {idx}: {', '.join(top_terms)}")

# Visualize topics
plt.figure(figsize=(10, 6))
topic_labels = [f"Topic {i}" for i in range(len(topics))]
word_counts = [np.sum(topic) for topic in lda.components_]
plt.bar(topic_labels, word_counts, color="lightcoral")
plt.xlabel("Topics")
plt.ylabel("Word Counts")
plt.title("Word Counts by Topic")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Associate topics with countries
print("\nAssociating Topics with Countries...")
printed_articles = set()
for idx, topic in enumerate(topics):
    print(f"Topic {idx}: {', '.join(topic)}")
    country_associations = {}

    for article in articles:
        if article in printed_articles:
            continue

        doc = nlp(article)
        for country, keywords in countries.items():
            if any(keyword in article for keyword in keywords):
                if country not in country_associations:
                    country_associations[country] = 0
                country_associations[country] += 1
        printed_articles.add(article)

    # Display country associations for the topic
    print("Associated Countries:")
    for country, count in sorted(country_associations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {country}: {count} articles")

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
nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="gray") # [data["weight"] for _, _, data in edges]
nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

# Add edge labels for weights
edge_labels = {(u, v): f"{data['weight']}" for u, v, data in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Country Co-occurrence Graph")
plt.show()