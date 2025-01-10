from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from datetime import datetime
import csv
import os

# Define base URL
base_url = "https://www.ecns.cn/news/politics"

# Define the cut-off date
cutoff_date = datetime.strptime("2025-01-01", "%Y-%m-%d")  # Example date

# Initialize the list to store news
news_list = []

# Set up Selenium WebDriver
options = Options()
options.add_argument("--headless")  # Run browser in headless mode
options.add_argument("--disable-gpu")  # Disable GPU for headless mode
options.add_argument("--no-sandbox")  # Required for some environments

service = Service("./chromedriver-mac-arm64/chromedriver")  # Update with your chromedriver path
driver = webdriver.Chrome(service=service, options=options)

# Folder to save articles
OUTPUT_FOLDER = "scraped_articles"

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def scrape_page(url):
    global news_list
    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.bizlst.dashedlne.mart5.overhid"))
        )

        articles_lists = driver.find_elements(By.CSS_SELECTOR, "ul.bizlst.dashedlne.mart5.overhid")

        if not articles_lists:
            print("No articles found on this page.")
            return False

        for articles_list in articles_lists:
            articles = articles_list.find_elements(By.TAG_NAME, "li")
            for article in articles:
                retry_count = 0
                while retry_count < 3:
                    try:
                        title_tag = article.find_element(By.TAG_NAME, "h3").find_element(By.TAG_NAME, "a")
                        title = title_tag.text.strip()
                        link = title_tag.get_attribute("href")
                        link = f"https:{link}" if link.startswith("//") else link

                        date_tag = article.find_element(By.TAG_NAME, "b")
                        date_text = date_tag.text.strip()
                        article_date = datetime.strptime(date_text, "%b %d, %Y")

                        if article_date < cutoff_date:
                            print("Reached cutoff date. Stopping scrape.")
                            return False

                        scrape_article_content(link, title)
                        news_list.append({"title": title, "link": link, "date": date_text})
                        break  # Successfully scraped, exit retry loop

                    except StaleElementReferenceException:
                        retry_count += 1
                        print(f"Stale element reference, retrying ({retry_count}/3)...")

                    except Exception as e:
                        print(f"Error parsing article: {e}")
                        break  # Exit retry loop on other exceptions
        return True

    except Exception as e:
        print(f"Error loading articles: {e}")
        return False

# Updated scraping function
def scrape_article_content(article_url, article_title):
    driver.get(article_url)

    try:
        # Wait for the article content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div#yanse.content"))
        )

        # Extract the content
        content_div = driver.find_element(By.CSS_SELECTOR, "div#yanse.content")
        paragraphs = content_div.find_elements(By.TAG_NAME, "p")
        content = "\n".join([p.text.strip() for p in paragraphs])

        if not content.strip():
            print(f"No content found for article: {article_title}")
            return

        # Save the content to a text file
        file_name = f"{article_title[:70].replace(' ', '_').replace('/', '_')}.txt"
        with open(os.path.join(OUTPUT_FOLDER, file_name), "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Saved article: {file_name}")

    except Exception as e:
        print(f"Error scraping article content from {article_url}: {e}")

def scrape_articles():
    article_links = driver.find_elements(By.CSS_SELECTOR, "div.article-list a")
    scraped_count = 0

    for link in article_links:
        try:
            # Re-locate the link element in each iteration
            article_title = link.text.strip()
            article_url = link.get_attribute("href")
            
            # Skip invalid links
            if not article_url or not article_title:
                continue

            scrape_article_content(article_url, article_title)
            scraped_count += 1
        except Exception as e:
            print(f"Error loading articles: {e}")

    print(f"Scraped {scraped_count} articles.")

def save_to_csv(filename="news.csv"):
    # Save scraped data to a CSV file.
    keys = ["title", "link", "date"]
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(news_list)

def scrape_all_pages():
    """Scrape articles across multiple pages."""
    page_number = 1
    while True:
        print(f"Scraping page {page_number}...")
        url = f"{base_url}/index_{page_number}.shtml" if page_number > 1 else f"{base_url}/index.shtml"
        continue_scraping = scrape_page(url)
        if not continue_scraping:
            break
        page_number += 1  # Move to the next page

# Main script execution
if __name__ == "__main__":
    try:
        scrape_all_pages()
        print(f"Scraped {len(news_list)} articles.")
        save_to_csv("ecns_politics_news.csv")
        print("Saved articles to ecns_politics_news.csv.")
    finally:
        driver.quit()  # Ensure the driver is closed properly