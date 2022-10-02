import requests
from bs4 import BeautifulSoup
import time
import re
import json

def get_categories():
    with open("../scrapes/categories_4chan.html") as f1:
        soup1 = BeautifulSoup(f1, 'html.parser')
        content1 = soup1.findAll("a", {"class":"boardlink"})
        categories = {
                        re.sub("//boards.4chan.org/|//boards.4channel.org/|/","",category.get('href')) : category.text 
                        for category in content1 if category.text != ""
                    }
    return categories

def get_content(categories, path, headers):
    with open(path, "r") as f2:
        data = json.load(f2)
        for acronym, category in categories.items():
            url = 'https://boards.4channel.org/{}'.format(acronym)
            time.sleep(5)
            r = requests.post(url, headers=headers)
            soup3 = BeautifulSoup(r.text, 'html.parser')
            corpus = soup3.findAll("blockquote")
    
            for content in corpus:
                try:
                    quote={}
                    quote["Category"] = category
                    quote["Acronym Category"] = acronym
                    quote["Reply"] = re.sub("[>>0-9]", "", content.text)
                except IndexError:
                    pass
                data.append(quote)
    return data

def store_content(path, data):
    with open(path, "w") as f3:
        json.dump(data, f3, indent=4, separators=(',',': '))
