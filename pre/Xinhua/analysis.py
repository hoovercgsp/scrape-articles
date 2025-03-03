from dataclasses import dataclass
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import re
import spacy
from nltk.tokenize import sent_tokenize
from utils import Article, Topic, infer_topic_from_keywords, separate_by_date

"""
INPUT
articles (List[Article]): A list of Article objects, each containing text content.
countries (Dict[str, List[str]], optional): A dictionary mapping country names to associated keywords (default: {}).

OUTPUT
Dict[Topic, List[Article]]: A dictionary where each key is a Topic, 
and the value is a list of Article objects assigned to that topic.
"""
def identify_topics_LDA(articles: List[Article], countries: Dict[str, List[str]] = {}):
    article_texts = [article.content for article in articles]
    vectorizer = CountVectorizer(stop_words="english", max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(article_texts)
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
    topic_distributions = lda.transform(X)
    topic_assignments = np.argmax(topic_distributions, axis=1)

    topic_dict = {Topic(title=infer_topic_from_keywords(topic), keywords=tuple(topic)): [] for topic in topics}
    for article, topic_idx in zip(articles, topic_assignments):
        topic_dict[Topic(title=infer_topic_from_keywords(topics[topic_idx]), keywords=tuple(topics[topic_idx]))].append(article)
    
    print(topic_dict.keys())

    return topic_dict

"""
INPUT
articles (List[Article]): A list of Article objects containing textual content.
topics (List[Topic]): A list of Topic objects.

OUTPUT
Dict[Topic, List[Article]]: A dictionary mapping topics (as tuples of keywords) to lists of articles associated with each topic.
"""
def categorize_by_topic(articles: List[Article], topics: List[Topic]) -> Dict[Topic, List[Article]]:
    documents = [article.content for article in articles]
    topic_keywords = list(set(keyword for topic in topics for keyword in topic.keywords))
    vectorizer = CountVectorizer(vocabulary=topic_keywords, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(n_components=len(topics), random_state=42)
    topic_distribution = lda.fit_transform(doc_term_matrix)

    category_map = defaultdict(list)
    for i, article in enumerate(articles):
        top_topic_index = np.argmax(topic_distribution[i])
        category_map[topics[top_topic_index]].append(article)
    
    return category_map

"""
INPUT
sentiment_model: A sentiment analysis model (e.g., Hugging Face pipeline).
nlp: A spaCy NLP model for text processing.
articles (List[Article]): A list of Article objects.
countries (Dict[str, List[str]]): A mapping of country names to lists of associated keywords.

OUTPUT
Dict[str, Dict[str, Counter]]: A dictionary where each country has a "total" count of analyzed sentences 
and a "scores" counter tracking sentiment labels.
"""
def sentiment_by_country(sentiment_model, nlp, articles: List[Article], countries: Dict[str, List[str]]):
    country_sentiment = {
        country: {"total": 0, "scores": Counter()} for country in countries
    }
    co_occurrence = Counter()

    for article in articles:
        doc = nlp(article.content)
        
        for paragraph in article.content.split("\n"):
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

"""
INPUT
articles (List[Article]): A list of Article objects.
sentiment_model: A sentiment analysis model.
topic: A topic to search for in articles.

OUTPUT
Dict[str, Counter]: A dictionary with topic as the key and a Counter object counting sentiment occurrences.
"""
def sentiment_by_keyword_topic(articles, sentiment_model, topic):

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

    def extract_relevant_sentences(articles, keywords):
        """
        Extract sentences from articles that contain any of the specified keywords.
        """
        relevant_sentences = []
        pattern = re.compile(r'\b(' + '|'.join([re.escape(kw) for kw in keywords]) + r')\b', re.IGNORECASE)
        
        for article in articles:
            sentences = sent_tokenize(article.content)  # Split article into sentences
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

    # Extract relevant sentences mentioning these keywords
    sentences = extract_relevant_sentences(articles, topic.keywords)
    
    # Perform sentiment analysis on extracted sentences
    sentiments = analyze_sentiment(sentences, sentiment_model)
    
    # Return results
    return {
        topic.title: Counter(sentiments)
    }

"""
INPUT
topic_articles (Topic, List[Article]]): A dictionary mapping topics to lists of articles.
sentiment_model: A sentiment analysis model.
tokenizer: A tokenizer for splitting text into chunks.
exclude_neutral (bool, optional): Whether to exclude neutral sentiment (default: True).

OUTPUT
Dict[Tuple[str, ...], Dict[str, int]]: A dictionary mapping topics to sentiment distributions.
"""
def sentiment_by_LDA_topic(topic_articles, sentiment_model, tokenizer, exclude_neutral=True):
    """
    Performs sentiment analysis on articles categorized by topics.
    """

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
        inputs = tokenizer(article.content, truncation=True, padding=True, max_length=512, return_tensors="pt")
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
    
    topic_sentiments = {}
    
    for topic, articles in topic_articles.items():
        sentiment_counts = Counter()
        
        for article in articles:
            sentiment_label = analyze_sentiment_chunked(article, sentiment_model, tokenizer, exclude_neutral)
            sentiment_counts[sentiment_label] += 1
        
        topic_sentiments[topic.title] = dict(sentiment_counts)
    
    return topic_sentiments

"""
INPUT
articles (List[Article]): A list of Article objects.
topic (Topic): A topic to analyze.

OUTPUT
Dict[str, Counter]: A dictionary mapping each keyword to a Counter of words appearing in its context.
"""
def discover_keyword_context(articles: List[Article], topic: Topic) -> Dict[str, Counter]:
    nlp = spacy.load("en_core_web_sm")
    keyword_contexts = {keyword: Counter() for keyword in topic.keywords}
    
    for article in articles:
        doc = nlp(article.content)
        for sentence in doc.sents:
            words = [token.text.lower() for token in sentence if token.is_alpha]
            for keyword in topic.keywords:
                if keyword in words:
                    index = words.index(keyword)
                    context_words = words[max(0, index - 5): index] + words[index + 1: index + 6]
                    keyword_contexts[keyword].update(context_words)
    
    return keyword_contexts

"""
INPUT
articles (List[Article]): A list of Article objects.
topic (Topic): A list of keywords (encapsulated in a Topic) to count occurrences of.

OUTPUT
Dict[str, int]: A dictionary mapping keywords to the number of times they appear in all articles.
"""
def num_keyword_mentions(articles: List[Article], topic: Topic) -> Dict[str, int]:
    keyword_counts = Counter()
    
    for article in articles:
        for keyword in topic.keywords:
            keyword_counts[keyword] += article.content.lower().count(keyword.lower())
    
    return keyword_counts
from collections import Counter
from typing import List, Dict, Tuple
import datetime


"""
BELOW: TIME SERIES FUNCTIONS, shadow functions for the four above
"""

def time_series_identify_topics_LDA(articles: List[Article], countries: Dict[str, List[str]] = {}):
    time_periods = separate_by_date(articles)
    return {period: identify_topics_LDA(time_periods[period], countries) for period in time_periods}

def time_series_categorize_by_topic(articles: List[Article], topics: List[Topic]) -> Dict[str, Dict[Topic, List[Article]]]:
    time_periods = separate_by_date(articles)
    return {period: categorize_by_topic(time_periods[period], topics) for period in time_periods}

def time_series_sentiment_by_country(sentiment_model, nlp, articles: List[Article], countries: Dict[str, List[str]]):
    time_periods = separate_by_date(articles)
    return {period: sentiment_by_country(sentiment_model, nlp, time_periods[period], countries) for period in time_periods}

def time_series_sentiment_by_keyword_topic(articles, sentiment_model, topic):
    time_periods = separate_by_date(articles)
    return {period: sentiment_by_keyword_topic(time_periods[period], sentiment_model, topic) for period in time_periods}

def time_series_sentiment_by_LDA_topic(topic_articles, sentiment_model, tokenizer, exclude_neutral=True):
    return {period: sentiment_by_LDA_topic(topic_articles[period], sentiment_model, tokenizer, exclude_neutral) for period in topic_articles}

def time_series_discover_keyword_context(articles: List[Article], topic: Topic) -> Dict[str, Dict[str, Counter]]:
    time_periods = separate_by_date(articles)
    return {period: discover_keyword_context(time_periods[period], topic) for period in time_periods}

def time_series_num_keyword_mentions(articles: List[Article], topic: Topic) -> Dict[str, Dict[str, int]]:
    time_periods = separate_by_date(articles)
    return {period: num_keyword_mentions(time_periods[period], topic) for period in time_periods}