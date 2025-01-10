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

# Define base URL
base_url = "https://www.ecns.cn/news/politics"

# Define the cut-off date
cutoff_date = datetime.strptime("2025-01-01", "%Y-%m-%d")

# Set up Selenium WebDriver
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

service = Service("./chromedriver-mac-arm64/chromedriver")  # Update with your chromedriver path
driver = webdriver.Chrome(service=service, options=options)

# Output folders and file
OUTPUT_FOLDER = "scraped_articles"
CSV_FILE = "articles.csv"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Helper functions
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def save_to_csv(data, filename):
    keys = ["title", "link", "date"]
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

def scrape_page(url):
    """Scrapes the given page for articles and collects data."""
    articles_data = []
    driver.get(url)

    try:
        # Wait for articles to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.bizlst.dashedlne.mart5.overhid"))
        )

        articles_lists = driver.find_elements(By.CSS_SELECTOR, "ul.bizlst.dashedlne.mart5.overhid")

        for articles_list in articles_lists:
            articles = articles_list.find_elements(By.TAG_NAME, "li")

            for article in articles:
                try:
                    # Extract article details
                    title_tag = article.find_element(By.TAG_NAME, "h3").find_element(By.TAG_NAME, "a")
                    title = title_tag.text.strip()
                    link = title_tag.get_attribute("href")
                    link = f"https:{link}" if link.startswith("//") else link

                    date_tag = article.find_element(By.TAG_NAME, "b")
                    date_text = date_tag.text.strip()
                    article_date = datetime.strptime(date_text, "%b %d, %Y")

                    # Filter by cutoff date
                    if article_date >= cutoff_date:
                        articles_data.append({"title": title, "link": link, "date": date_text})
                    else:
                        return articles_data  # Stop processing older articles

                except Exception as e:
                    print(f"Error processing article: {e}")

        return articles_data

    except Exception as e:
        print(f"Error scraping page {url}: {e}")
        return []

def scrape_article_content(article):
    """Scrapes content of an article and saves it as a .txt file."""
    article_url = article["link"]
    article_title = article["title"]

    driver.get(article_url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div#yanse.content"))
        )

        content_div = driver.find_element(By.CSS_SELECTOR, "div#yanse.content")
        paragraphs = content_div.find_elements(By.TAG_NAME, "p")
        content = "\n".join([p.text.strip() for p in paragraphs])

        if content.strip():
            file_name = sanitize_filename(f"{article_title[:70].replace(' ', '_')}.txt")
            with open(os.path.join(OUTPUT_FOLDER, file_name), "w", encoding="utf-8") as file:
                file.write(content)
            print(f"Saved article content: {file_name}")

    except Exception as e:
        print(f"Error scraping content for {article_url}: {e}")

# Main execution
def main():
    try:
        all_articles = []
        page_number = 1

        while True:
            print(f"Scraping page {page_number}...")
            url = f"{base_url}/index_{page_number}.shtml" if page_number > 1 else f"{base_url}/index.shtml"
            articles = scrape_page(url)

            if not articles:
                break

            all_articles.extend(articles)
            page_number += 1

        # Save articles to CSV
        save_to_csv(all_articles, CSV_FILE)
        print(f"Saved {len(all_articles)} articles to {CSV_FILE}.")

        # Scrape content for each article
        for article in all_articles:
            scrape_article_content(article)

    finally:
        driver.quit()

if __name__ == "__main__":
    main()