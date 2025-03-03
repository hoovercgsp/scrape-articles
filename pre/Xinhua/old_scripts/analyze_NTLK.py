import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, RegexpParser
from collections import Counter
import nltk
from itertools import combinations
from collections import defaultdict
from gensim import corpora, models
import networkx as nx
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download('punkt_tab')

def calculate_cooccurrences(texts, window_size=5):
    """
    Calculates word co-occurrences within a specified window size.
    Args:
        texts (list): List of preprocessed token lists, one for each article.
        window_size (int): The size of the sliding window for co-occurrence.
    Returns:
        dict: A dictionary where keys are word pairs and values are their co-occurrence frequencies.
    """
    cooccurrence_counts = defaultdict(int)

    for tokens in texts:
        for i, word in enumerate(tokens):
            # Define the sliding window
            window_start = max(0, i - window_size)
            window_end = min(len(tokens), i + window_size + 1)
            window = tokens[window_start:window_end]

            # Create word pairs within the window and count them
            for other_word in window:
                if word != other_word:
                    pair = tuple(sorted([word, other_word]))
                    cooccurrence_counts[pair] += 1

    return cooccurrence_counts

def analyze_word_cooccurrences(contents, window_size=5):
    """
    Analyzes word co-occurrences in the aggregated content of articles.
    Args:
        contents (dict): A dictionary where keys are article titles and values are file contents.
        window_size (int): The size of the sliding window for co-occurrence.
    Returns:
        None
    """
    # Preprocess each article separately
    processed_texts = [preprocess_text(content) for content in contents.values()]

    # Calculate co-occurrences
    cooccurrence_counts = calculate_cooccurrences(processed_texts, window_size)

    # Sort by frequency and display results
    sorted_cooccurrences = sorted(cooccurrence_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 10 word co-occurrences (within a window of {window_size}):")
    for pair, count in sorted_cooccurrences[:20]:
        print(f"{pair}: {count}")

def preprocess_text(text):
    """
    Preprocesses text by tokenizing, removing stop words, applying stemming and lemmatization.

    Args:
        text (str): The input text.

    Returns:
        list: A list of processed tokens.
    """
    # Tokenize text
    tokens = word_tokenize(text)

    # Convert to lowercase and remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

    # Apply stemming and lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    processed_tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens]

    return processed_tokens

def analyze_articles(contents):
    """
    Analyzes the aggregated content of articles.
    Args:
        contents (dict): A dictionary where keys are article titles and values are the file contents.
    Returns:
        None
    """
    # Combine all text from articles into one string
    aggregate_text = " ".join(contents.values())

    # Preprocess the aggregated text
    processed_tokens = preprocess_text(aggregate_text)

    # Calculate word frequencies
    word_freq = Counter(processed_tokens)
    print("Most common words:")
    print(word_freq.most_common(20))
    
    # # Perform POS tagging
    # pos_tags = pos_tag(processed_tokens)
    # print("\nSample POS tagging:")
    # print(pos_tags[:10])

    # # Chunking example to detect patterns (e.g., Noun Phrases)
    # grammar = "NP: {<DT>?<JJ>*<NN>}"  # Define a simple grammar for noun phrases
    # chunk_parser = RegexpParser(grammar)
    # tree = chunk_parser.parse(pos_tags)
    # print("\nChunking example (first few chunks):")
    # print(tree[:10])

    # # Chinking example: Exclude verbs from noun phrases
    # grammar_with_chinking = r"""
    # NP: {<DT>?<JJ>*<NN>}  # Chunk determiner/adjective/noun
    #     }<VB.*>{          # Exclude verbs from chunks
    # """
    # chunk_parser_with_chinking = RegexpParser(grammar_with_chinking)
    # tree_with_chinking = chunk_parser_with_chinking.parse(pos_tags)
    # print("\nChinking example (first few chunks):")
    # print(tree_with_chinking[:10])

def read_text_files(folder_path):
    """
    Reads all .txt files in the specified folder and returns their contents as a dictionary.
    Args:
        folder_path (str): Path to the folder containing the .txt files.
    Returns:
        dict: A dictionary where keys are article titles and values are the file contents including the title.
    """
    try:
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Dictionary to store file contents
        file_contents = {}

        # Iterate over all files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):  # Process only .txt files
                file_path = os.path.join(folder_path, file_name)
                
                # Read the content of the text file
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    if lines and lines[0].startswith("Title: "):
                        article_title = lines[0].replace("Title: ", "").strip()
                        file_contents[article_title] = "".join(lines)
                    else:
                        file_contents[file_name] = "".join(lines)

        return file_contents

    except Exception as e:
        print(f"Error: {e}")
        return {}

def build_cooccurrence_graph(articles, window_size=5):
    graph = nx.Graph()
    all_words = []
    
    # Collect words from all articles
    for article in articles:
        tokens = word_tokenize(article.lower())
        filtered_tokens = [word for word in tokens if word.isalnum()]
        all_words.extend(filtered_tokens)
        
        # Create edges based on co-occurrence within a sliding window
        for i in range(len(filtered_tokens) - window_size + 1):
            window = filtered_tokens[i:i+window_size]
            for w1, w2 in combinations(window, 2):
                graph.add_edge(w1, w2, weight=graph.get_edge_data(w1, w2, default={'weight': 0})['weight'] + 1)
    
    return graph

def train_word2vec(processed_articles, vector_size=100, window=5, min_count=2, workers=4):
    model = Word2Vec(sentences=processed_articles, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

def get_documents_for_topics(ldamodel, corpus, num_topics=5):
    """
    Extracts documents associated with each LDA topic.
    :param ldamodel: The trained LDA model.
    :param corpus: The bag-of-words corpus of the documents.
    :param num_topics: The number of topics to display.
    :return: Dictionary where each key is a topic and the value is a list of document indices.
    """
    # Create a dictionary for storing the topic to document mappings
    topic_docs = {i: [] for i in range(num_topics)}
    
    # Loop over each document and get the most likely topic
    for doc_index, doc_bow in enumerate(corpus):
        topic_dist = ldamodel.get_document_topics(doc_bow)
        # Get the topic with the highest probability
        best_topic = max(topic_dist, key=lambda x: x[1])[0]
        topic_docs[best_topic].append(doc_index)
    
    return topic_docs

def print_top_words_for_topic(ldamodel, num_topics=5, top_n=10):
    """
    Prints the top words for each LDA topic.
    :param ldamodel: The trained LDA model.
    :param num_topics: The number of topics to display.
    :param top_n: The number of top words to display for each topic.
    """
    for topic_id in range(num_topics):
        print(f"Topic {topic_id}:")
        words = ldamodel.show_topic(topic_id, topn=top_n)
        for word, prob in words:
            print(f"  {word}: {prob}")
        print()

def plot_cooccurrence_graph(edges, threshold=50, top_n=25):
    # Count the occurrences of each word in the edges
    word_counts = Counter()
    for edge in edges:
        word1, word2, data = edge
        word_counts[word1] += data.get('weight', 0)
        word_counts[word2] += data.get('weight', 0)

    # Get the top 'n' most common words
    top_words = [word for word, _ in word_counts.most_common(top_n)]

    # Create an empty graph
    G = nx.Graph()

    # Add edges with their corresponding weights (count or co-occurrence frequency)
    for edge in edges:
        word1, word2, data = edge
        count = data.get('weight', 0)  # Make sure 'weight' is the right attribute for co-occurrence count

        # Check if both words are in the top words list and the count exceeds the threshold
        if word1 in top_words and word2 in top_words and count >= threshold:
            G.add_edge(word1, word2, weight=count)

    # Get edge weights for visualization
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize edge thickness
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    edge_thickness = [(weight - min_weight) / (max_weight - min_weight) * 5 + 1 for weight in edge_weights]  # Scaled thickness
    
    # Draw the graph with edge thickness and color
    pos = nx.spring_layout(G, k=0.15, iterations=20)  # layout for better visualization
    plt.figure(figsize=(12, 12))  # Increase figure size for clarity
    
    # Color edges based on weight (lighter colors for lower weights)
    edge_colors = [G[u][v]['weight'] / max_weight for u, v in G.edges()]  # Color edges based on weight
    
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=edge_thickness, edge_color=edge_colors, edge_cmap=plt.cm.Blues, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.title("Co-occurrence Graph of Top 25 Most Frequent Words")
    plt.axis('off')  # Hide axis
    plt.show()

if __name__ == "__main__":
    # Define the path to the folder containing the .txt files
    folder_path = os.path.join(os.path.dirname(__file__), "articles_asia_pacific")

    # Extract the text from all files
    txt_files_content = read_text_files(folder_path)

    # Preprocess text files into tokenized lists
    processed_articles = [preprocess_text(content) for content in txt_files_content.values()]

    # OPTION 0: Manual Analysis of word frequency and co-occurrence
    analyze_articles(txt_files_content)
    analyze_word_cooccurrences(txt_files_content, window_size=5)

    # OPTION 1: LDA (Latent Dirichlet Allocation), topic modeling
    dictionary = corpora.Dictionary(processed_articles)  # Use tokenized articles
    corpus = [dictionary.doc2bow(article) for article in processed_articles]  # Bag-of-words representation

    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    print("\nLDA Topics:")
    for idx, topic in lda_model.print_topics():
        print(f"Topic {idx}: {topic}")

    # Explore topics further by printing the top words for each topic
    print("\nDetailed Topic Analysis:")
    print_top_words_for_topic(lda_model)

    # Get documents associated with each topic
    topic_docs = get_documents_for_topics(lda_model, corpus, num_topics=5)
    print("\nDocuments Associated with Each Topic:")
    for topic_id, doc_indices in topic_docs.items():
        print(f"Topic {topic_id} documents:")

    # OPTION 2: Co-occurrence graph
    cooccurrence_graph = build_cooccurrence_graph([" ".join(article) for article in processed_articles])
    print("\nCo-occurrence Graph:")
    print("Number of nodes:", cooccurrence_graph.number_of_nodes())
    print("Number of edges:", cooccurrence_graph.number_of_edges())
    nx.write_gexf(cooccurrence_graph, "cooccurrence_graph.gexf")

    # Visualize the co-occurrence graph
    print("\nVisualizing Co-occurrence Graph...")
    plot_cooccurrence_graph(cooccurrence_graph.edges(data=True), threshold=50)

    # OPTION 3: Word embeddings
    word2vec_model = train_word2vec(processed_articles)
    similar_words = word2vec_model.wv.most_similar("china", topn=10)
    print("\nWords similar to 'china':", similar_words)


