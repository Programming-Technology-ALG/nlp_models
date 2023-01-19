import json
import re
import warnings
from pathlib import Path

import pandas as pd
import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

warnings.filterwarnings('ignore')

PAGES_LOAD = 30
CONSOLE = True

cache_dir =  Path('cache/')
cache_dir.mkdir(exist_ok=True)


category_links = [
    ('https://www.nn.ru/text/?rubric=economics&page=', 'economics'),
    ('https://www.nn.ru/text/?rubric=gorod&page=', 'gorod'),
    ('https://www.nn.ru/text/?rubric=culture&page=', 'culture'),
    ('https://www.nn.ru/text/?rubric=incidents&page=', 'incidents'),
    ('https://www.nn.ru/text/?rubric=politics&page=', 'politics'),
    ('https://www.nn.ru/text/?rubric=world&page=', 'world')
]


chrome_options = Options()
prefs = {"profile.managed_default_content_settings.images": 2,
            "profile.default_content_settings.cookies": 2,
            "profile.default_content_setting_values.notifications" : 2}
chrome_options.add_experimental_option("prefs",prefs)
chrome_options.add_argument('--disable-application-cache')

driver = webdriver.Chrome(options=chrome_options)


link_path = cache_dir / f'links_{PAGES_LOAD}.json'

if not link_path.exists():
    category_res = {}
    for link, category in category_links:
        news_links = []
        if CONSOLE:
            print(f'Processing {category.upper()}', flush=True)
        for page_id in tqdm.tqdm(range(1, PAGES_LOAD + 1), desc=category.upper(), disable=CONSOLE):

            page_link = link + str(page_id)
            driver.get(page_link)
            page_news = driver.find_elements(By.CLASS_NAME, "h9Jmx")
            
            if len(page_news) == 0: break
            for element in page_news:
                news_links.append(element.find_element(By.TAG_NAME, 'a').get_attribute('href'))
            print(page_link, len(news_links), flush=True)
        category_res[category] = news_links
    driver.quit()

    data = json.dumps(category_res)
    with open(link_path, "w") as f:
        f.write(data)
else:
    print('Used cached link file')

with open(link_path, "r") as f:
    links = json.load(f)

leng = []
total = 0
for values in list(links.values()):
    len_values = len(values)
    leng.append(len_values)
    total += len_values
print(leng, total, flush=True)


news_path = cache_dir / f'news_{total}.json'
if not news_path.exists():
    news = []
    driver = webdriver.Chrome(chrome_options=chrome_options)
    for category in links:
        if CONSOLE:
            print(f'Processing {category.upper()}', flush=True)
        for link in tqdm.tqdm(links[category], desc=category.upper(), disable=CONSOLE):
            try:
                driver.get(link)
                page_text_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'qQq9J')]")
                raw_text = " ".join([i.get_property('innerText') for i in page_text_elements]).replace('\n', " ")
                text = re.sub(" +", " ", raw_text)
                key_words = driver.find_element(By.XPATH, "//meta[contains(@name, 'news_keywords')]").get_attribute('content')
                title = driver.find_element(By.XPATH, "//h1[contains(@itemprop, 'headline')]").get_property('innerText')
                news.append({
                        'article_id' : link,
                        'title': title,
                        'category': category,
                        'tags': key_words,
                        'text': text
                    })
            except:
                print('Error', flush=True)
                continue
    driver.quit()

    with open(news_path, "w", encoding='UTF-8') as f:
        f.write(json.dumps({
            'catalog': news
        }, ensure_ascii=False, indent=4))
else:
    print('Used cached news file')

with open(news_path, 'r', encoding="UTF-8") as f:
    news = pd.DataFrame(json.load(f).get('catalog'))
    print(news, flush=True)
