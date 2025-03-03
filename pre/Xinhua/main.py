import utils
import analysis
import visualization
from utils import Topic, Article

def sample_time_series():
    # Define the topic to analyze
    topic = "in_depth"

    # Load articles related to the topic
    articles = utils.get_articles([topic])

    if not articles:
        print(f"No articles found for topic: {topic}")
        exit()

    # Load models
    sentiment_model, nlp, tokenizer, w2v_model = utils.load_models(articles)

    print(f"Loaded {len(articles)} articles on {topic}.")

    sample_topic = Topic("trade", ["development", "trade", "investment", "China", "United States"])

    # Separate articles by time period
    time_periods = utils.separate_by_date(articles)

    # Identify topics using LDA over time
    time_topic_dicts = analysis.time_series_identify_topics_LDA(time_periods)
    print("Identified topics for each time period.")

    # Sentiment analysis by topic over time
    time_topic_sentiments = analysis.time_series_sentiment_by_LDA_topic(time_topic_dicts, sentiment_model, tokenizer)
    print("Sentiment analysis by topic over time complete.")
    
    # Pass formatted time-topic sentiment data to visualization
    visualization.plot_time_series_sentiment_by_category(time_topic_sentiments, title="Sentiment by topic over time")

    # Count keyword mentions over time
    time_keyword_counts = analysis.time_series_num_keyword_mentions(time_periods, sample_topic)
    print("Keyword mentions over time:", time_keyword_counts)

    # Extract keyword context over time
    time_keyword_contexts = analysis.time_series_discover_keyword_context(time_periods, sample_topic)
    print("Keyword contexts extracted over time.")
    
    # Pass extracted keyword context data to visualization
    visualization.plot_time_series_keyword_context(time_keyword_contexts, keyword="trade")

    # Print and visualize topic distribution over time
    for period, topic_dict in time_topic_dicts.items():
        print(f"\nTime Period: {period}")
        print(f"Sample topic breakdown:")
        for topic, topic_articles in list(topic_dict.items())[:3]:
            print(f"Topic: {topic.title} - {len(topic_articles)} articles")
        
        # Visualize topic distribution
        visualization.plot_topic_distribution(topic_dict, title=f"Topic Distribution ({period})")

    print("\nSample topic sentiments over time:")
    for period, topic_sentiments in time_topic_sentiments.items():
        print(f"\nTime Period: {period}")
        for topic, sentiments in list(topic_sentiments.items())[:3]:
            print(f"Topic: {', '.join(topic)} - Sentiments: {sentiments}")
        
        # Visualize sentiment comparison
        visualization.plot_compare_sentiments(topic_sentiments, title=f"Sentiment Comparison ({period})")

    print("\nKeyword counts over time:")
    for period, keyword_counts in time_keyword_counts.items():
        print(f"\nTime Period: {period}")
        for keyword, count in keyword_counts.items():
            print(f"{keyword}: {count}")

    print("\nKeyword sentiments over time:")
    time_keyword_sentiments = analysis.time_series_sentiment_by_keyword_topic(time_periods, sentiment_model, sample_topic)
    
    for period, keyword_sentiments in time_keyword_sentiments.items():
        # Ensure correct data structure for visualization
        visualization.plot_time_series_sentiment_by_category(keyword_sentiments, title=f"Sentiment by keyword group ({period})")

def sample():
    # Define the topic to analyze
    topic = "in_depth"
    
    # Load articles related to the topic
    articles = utils.get_articles([topic])

    if not articles:
        print(f"No articles found for topic: {topic}")
        return

    # Load models
    sentiment_model, nlp, tokenizer, w2v_model = utils.load_models(articles)

    print(f"Loaded {len(articles)} articles on {topic}.")

    sample_topic = Topic("trade", ["development", "trade", "investment", "China", "United States"])

    # Identify topics using LDA
    topic_dict = analysis.identify_topics_LDA(articles)
    print(f"Identified {len(topic_dict)} topics.")

    # Sentiment analysis by topic
    topic_sentiments = analysis.sentiment_by_LDA_topic(topic_dict, sentiment_model, tokenizer)
    print("Sentiment analysis by topic complete.")
    visualization.plot_sentiment_by_category(topic_sentiments, "Sentiment by topic")

    # Count keyword mentions
    keyword_counts = analysis.num_keyword_mentions(articles, sample_topic)
    print("Keyword mentions:", keyword_counts)

    # Extract keyword context
    keyword_contexts = analysis.discover_keyword_context(articles, sample_topic)
    print("Keyword contexts extracted.")
    visualization.plot_keyword_context(keyword_contexts, "trade")

    # Print sample results
    print("\nSample topic breakdown:")
    for topic, topic_articles in list(topic_dict.items())[:3]:
        print(f"Topic: {topic.title} - {len(topic_articles)} articles")
    visualization.plot_topic_distribution(topic_dict, "Topic Distribution")

    print("\nSample topic sentiments:")
    for topic, sentiments in list(topic_sentiments.items())[:3]:
        print(f"Topic: {', '.join(topic)} - Sentiments: {sentiments}")
    print(topic_sentiments)
    visualization.plot_compare_sentiments(topic_sentiments, "Sentiment Comparison")

    print("\nKeyword counts:")
    for keyword, count in keyword_counts.items():
        print(f"{keyword}: {count}")

    print("\nKeyword sentiments:")
    keyword_sentiments = analysis.sentiment_by_keyword_topic(articles, sentiment_model, sample_topic)
    visualization.plot_sentiment_by_category(keyword_sentiments, "Sentiment by keyword group")


def main():
    sample_time_series()


if __name__ == "__main__":
    main()