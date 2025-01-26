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
cutoff_date = datetime.strptime("2025-01-20", "%Y-%m-%d")
keywords = []

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

def scrape_page():
    """Scrapes the current page for articles and collects data."""
    articles_data = []

    # Wait for articles to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "item"))
    )

    articles = driver.find_elements(By.CLASS_NAME, "item")
    for article in articles:
        try:
            title_element = article.find_element(By.CLASS_NAME, "tit").find_element(By.TAG_NAME, "a")
            date_element = article.find_element(By.CLASS_NAME, "time")

            title = title_element.text
            link = title_element.get_attribute("href")
            date_text = date_element.text.strip()

            # Validate date_text and try to parse
            if date_text:
                try:
                    date = datetime.strptime(date_text, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    print(f"Date format error for article '{title}': {date_text}")
                    continue  # Skip articles with unrecognized date formats
            else:
                if (len(title) > 0):
                    print(f"No date found for article '{title}'. Skipping...")
                continue

            # Filter articles by cutoff date
            if date >= cutoff_date:
                print(f"Scraped Article: {title}")
                articles_data.append({
                    "title": title,
                    "link": link,
                    "date": date_text
                })
        except Exception as e:
            print(f"Error processing article: {e}")

    return articles_data


def click_more_button():
    """Clicks the 'More' button to load additional articles."""
    try:
        more_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "more"))
        )
        more_button.click()
        # Wait for new articles to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "item"))
        )
        print("Loaded more articles...")
        return True
    except Exception as e:
        print(f"No more articles to load or error clicking 'More': {e}")
        return False

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

        while True:
            articles = scrape_page()
            all_articles.extend(articles)

            # Attempt to click the 'More' button to load additional articles
            if not click_more_button():
                break

        # Save articles to CSV
        save_to_csv(all_articles, csv_file)

        # Scrape content for each article
        for article in all_articles:
            scrape_article_content(article, output_folder)

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
