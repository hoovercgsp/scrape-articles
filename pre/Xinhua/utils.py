import datetime
import os
import re
from dataclasses import dataclass
import torch
import spacy
from transformers import pipeline, AutoTokenizer
from gensim.models import Word2Vec, KeyedVectors
# python -m spacy download en_core_web_md

@dataclass
class Article:
    title: str
    date: datetime.datetime
    content: str

@dataclass(frozen=True)
class Topic:
    title: str
    keywords: tuple[str, ...] 

SENTIMENT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

EMBEDDING_MODEL_PATH = "word2vec_xinhua.model"

def infer_topic_from_keywords(keywords):
    """Infer topic using the most similar word in spaCy's vocabulary."""
    keyword_tokens = [token for token in nlp.pipe(keywords) if token.has_vector]

    if not keyword_tokens:
        return "Unknown Topic"  # No valid words with vectors

    # Extract vectors for all tokens
    keyword_vectors = np.array([token.vector for token in keyword_tokens])

    # Compute pairwise similarities
    similarity_sums = np.dot(keyword_vectors, keyword_vectors.T).sum(axis=1)

    # Select the most representative word
    most_representative_idx = np.argmax(similarity_sums)
    return keyword_tokens[most_representative_idx].text

def load_models(articles):
    def train_word2vec(articles):
        """
        Trains a Word2Vec model on the provided articles.
        Returns the trained model.
        """
        # Tokenize articles into words (basic preprocessing)
        sentences = [re.findall(r'\b\w+\b', article.lower()) for article in articles]
        
        # Train Word2Vec model
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
        
        return model
    
    sentiment_model = pipeline(
        "text-classification", 
        model=SENTIMENT_MODEL_NAME,
        device = 0 if torch.cuda.is_available() else -1,
        top_k = None
    )
    nlp = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)

    print("Training new Word2Vec model...")
    w2v_model = train_word2vec([article.content for article in articles])
    w2v_model.save(EMBEDDING_MODEL_PATH)

    return sentiment_model, nlp, tokenizer, w2v_model

def parse_filename(filename):
    """Extracts the date and title from the filename."""
    date_str, title_with_extension = filename.split("_", 1)
    title = title_with_extension.rsplit(".", 1)[0]
    try:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ValueError(f"Date format in filename {filename} is incorrect.")
    return date, title

def get_articles(topics):
    articles = []
    
    for topic in topics:
        folder_path = os.path.join("scraped_content", f"articles_{topic}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    date, title = parse_filename(filename)
                    articles.append(Article(title=title, date=date, content=content))
    
    articles.sort(key=lambda article: article.date)
    
    return articles

def separate_by_date(articles):
    """Separates articles into four time eras."""
    time_periods = {
        "before_2023_12_31": [],
        "between_2024_01_01_and_2024_11_05": [],
        "between_2024_11_05_and_2025_01_20": [],
        "after_2025_01_20": []
    }
    
    date_2023_12_31 = datetime.datetime(2023, 12, 31)
    date_2024_01_01 = datetime.datetime(2024, 1, 1)
    date_2024_11_05 = datetime.datetime(2024, 11, 5)
    date_2025_01_20 = datetime.datetime(2025, 1, 20)
    
    for article in articles:
        if article.date <= date_2023_12_31:
            time_periods["before_2023_12_31"].append(article)
        elif date_2024_01_01 <= article.date <= date_2024_11_05:
            time_periods["between_2024_01_01_and_2024_11_05"].append(article)
        elif date_2024_11_05 < article.date <= date_2025_01_20:
            time_periods["between_2024_11_05_and_2025_01_20"].append(article)
        else:
            time_periods["after_2025_01_20"].append(article)
    
    return time_periods

def identify_event_coverage(articles, keywords, start_date, end_date):
    """Finds articles that match keywords and fall within the date range."""
    relevant_articles = []
    
    for article in articles:
        if start_date <= article.date <= end_date:
            if any(keyword.lower() in article.title.lower() or keyword.lower() in article.content.lower() for keyword in keywords):
                relevant_articles.append(article)
    
    return relevant_articles

def preprocess_articles(articles, countries):
    """ Remove country names and demonyms from articles, preserving punctuation. """
    country_words = set(word.lower() for country in countries.values() for word in country)
    
    def clean_article(article):
        return " ".join(
            word for word in re.findall(r"\b\w+\b", article) if word.lower() not in country_words
        )
    
    return [clean_article(article) for article in articles]