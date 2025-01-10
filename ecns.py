from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import csv

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

service = Service("/Users/ejw675/Desktop/LD/chromedriver-mac-arm64/chromedriver")  # Update with your chromedriver path
driver = webdriver.Chrome(service=service, options=options)

# Folder to save articles
OUTPUT_FOLDER = "scraped_articles"

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def scrape_page(url):
    global news_list
    driver.get(url)

    try:
        # Wait until at least one article list appears
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.bizlst.dashedlne.mart5.overhid"))
        )

        # Find all article lists
        articles_lists = driver.find_elements(By.CSS_SELECTOR, "ul.bizlst.dashedlne.mart5.overhid")

        if not articles_lists:
            print("No articles found on this page.")
            return False

        for articles_list in articles_lists:
            articles = articles_list.find_elements(By.TAG_NAME, "li")
            for article in articles:
                try:
                    # Extract title and link
                    title_tag = article.find_element(By.TAG_NAME, "h3").find_element(By.TAG_NAME, "a")
                    title = title_tag.text.strip()
                    link = title_tag.get_attribute("href")
                    link = f"https:{link}" if link.startswith("//") else link

                    # Extract date
                    date_tag = article.find_element(By.TAG_NAME, "b")
                    date_text = date_tag.text.strip()

                    # Parse date
                    article_date = datetime.strptime(date_text, "%b %d, %Y")
                    if article_date < cutoff_date:
                        print("Reached cutoff date. Stopping scrape.")
                        return False

                    # Append to global news list
                    news_list.append({"title": title, "link": link, "date": date_text})
                except Exception as e:
                    print(f"Error parsing article: {e}")
        return True

    except Exception as e:
        print(f"Error loading articles: {e}")
        print(driver.page_source)  # Debug: Print page content for analysis
        return False

def save_to_csv(filename="news.csv"):
    """Save scraped data to a CSV file."""
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