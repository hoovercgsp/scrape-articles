import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from datetime import datetime
import csv
import os
import re
import time

# Define base URL; currently politics-related articles
africa_base_url = "https://english.news.cn/africa/china_africa/index.htm"

# Define the cut-off date and keywords
cutoff_date = datetime.strptime("2025-01-01", "%Y-%m-%d")
keywords = []
scraped_links = set()
done = False

# Set up Selenium WebDriver
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

service = Service("../chromedriver-mac-arm64/chromedriver")  # Update with your chromedriver path
driver = webdriver.Chrome(service=service, options=options)

# Helper functions
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def save_to_csv(data, filename):
    keys = ["title", "link", "date"]
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

def click_more_button(driver):
    """Clicks the 'More' button to load more articles, handling structure and overlap."""
    try:
        # Wait for the "More" button to be present and visible
        more_button = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "more"))
        )
        
        # Ensure it's in view
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_button)
        time.sleep(1)  # Ensure scrolling completes

        # Click the button using JavaScript
        driver.execute_script("arguments[0].click();", more_button)
        print("Clicked 'More' button to load more articles.")
        
        # Wait for new articles to be loaded in the `list-cont` container
        WebDriverWait(driver, 10).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, ".list-cont > *")) > 0
        )
    except Exception as e:
        print(f"Error clicking 'More' button: {e}")

def wait_for_new_articles(prev_article_count):
    """Waits for new articles to load by checking for an increase in article count."""
    try:
        WebDriverWait(driver, 10).until(
            lambda d: len(d.find_elements(By.CLASS_NAME, "item")) > prev_article_count
        )
        print("New articles loaded.")
    except Exception as e:
        print(f"Error waiting for new articles: {e}")

def scrape_page():
    """Scrapes the current page for articles and collects data."""
    articles_data = []
    
    # Get the current number of articles
    articles = driver.find_elements(By.CLASS_NAME, "item")
    prev_article_count = len(articles)

    for article in articles:
        try:
            title_element = article.find_element(By.CLASS_NAME, "tit").find_element(By.TAG_NAME, "a")
            link = title_element.get_attribute("href")
            title = title_element.text
            
            time_element = article.find_element(By.CLASS_NAME, "time")
            date_text = time_element.text
            date = datetime.strptime(date_text, "%Y-%m-%d %H:%M:%S")

            if date < cutoff_date:
                print(f"Stopping scraping: Found article with date {date} before cut-off date {cutoff_date}")
                done = True
                return articles_data

            # Skip duplicates
            if link in scraped_links:
                continue
            
            scraped_links.add(link)  # Track the link to avoid re-scraping
            print(f"Scraped Article: {title}")

            articles_data.append({
                "title": title,
                "link": link,
                "date": date
            })
        except Exception as e:
            print(f"Error processing article: {e}")

    return articles_data

def scrape_all_pages(driver):
    """Scrapes all pages by clicking 'More' and scraping new articles."""
    all_articles = []

    while True:
        current_articles = scrape_page()
        all_articles.extend(current_articles)

        if done:
            return all_articles

        prev_article_count = len(scraped_links)
        click_more_button(driver)  # Click the 'More' button
        time.sleep(2)  # Give time for the page to react

        wait_for_new_articles(prev_article_count)

    return all_articles

def scrape_article_content(article, output_folder):
    """Scrapes content of an article and saves it as a .txt file."""
    article_url = article["link"]
    article_title = sanitize_filename(article["title"])
    article_date = article["date"]

    driver.get(article_url)

    try:
        # Wait for the article content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        content_elements = driver.find_elements(By.TAG_NAME, "p")
        content = "\n".join([element.text for element in content_elements if element.text])

        # Save content to a file
        file_path = os.path.join(output_folder, f"{article_date}_{article_title}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Saved article: {article_title}")
    except Exception as e:
        print(f"Error scraping article content: {e}")

# Main execution
def main():
    parser = argparse.ArgumentParser(description="Scrape articles and their content.")
    parser.add_argument("--csv_file", type=str, default="articles.csv", help="Name of the CSV file to save article data.")
    parser.add_argument("--output_folder", type=str, default="scraped_articles", help="Name of the folder to save article content.")

    args = parser.parse_args()

    csv_file = args.csv_file
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    try:
        driver.get(africa_base_url)

        all_articles = []  # each article has "link", "date", and "title"

        all_articles = scrape_all_pages(driver)

        # Save articles to CSV
        save_to_csv(all_articles, csv_file)

        # Scrape content for each article
        for article in all_articles:
            scrape_article_content(article, output_folder)

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
