In order to run this script, clone this repo on your local computer. 
Replace the chromedriver-mac-arm64 folder with the version of chrome driver which matches your device here: https://developer.chrome.com/docs/chromedriver/downloads. You can check your chrome version at chrome://settings/help. The folder currently in this repo works for chrome version 131.0.6778.265 on Macbooks released November 2020 or later. 
Selenium (the scraping tool) should already be downloaded in the local environment provided. 

In order to scrape the articles, navigate to the directory containing the repo. You may edit lines 18-20 in ecsn.py to change the cutoff date for articles that you want to scrape, and scrape based on the presence of given keywords in the title of the article. You may also change the base URL on line 16, but ensure that the website is from ECSN and has the same format as the politics page currently there.

After you've made your edits, run:
python ecns.py --csv_file <name_of_csv_file.csv> --output_folder <name_of_folder_containing_txt_files>

For example:
python ecns.py --csv_file politics_in_2024.csv --output_folder politics_in_2024

Currently, the politics_in_2024.csv file contains a list of all the articles released on https://www.ecns.cn/news/politics in 2024. The folder of the same name contains text files of the content of those articles.