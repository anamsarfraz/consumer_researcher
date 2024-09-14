import requests
from bs4 import BeautifulSoup
import re

def parse_google_search_results(text, max_results=3):
    links = []
    ## split then join to convert spaces to + in link
    url = 'https://google.com/search?q=' + '+'.join(text.split())
    response_content = requests.get(url).content
    soup = BeautifulSoup(response_content, "html.parser")

    ## loop through only the first results up to max_results
    for d in soup.select('div:has(>div>a[href] h3)')[:max_results]:
        title = d.h3.get_text(' ').strip()
        

        ## link
        res_link = d.select_one('a[href]:has(h3)').get('href') 
        if res_link.startswith('/url?q='):
            res_link = res_link.split('=',1)[1].split('&')[0]
        
        links.append(res_link)
        description = d.select_one('div:has(>a[href] h3)+div').get_text(' ').strip()
        
        #print(f'From {url}\n---\n{title}\n{res_link}\n{description}\n---\n')
    return links
  
def parse_webpage_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = [p.text for p in soup.find_all("p")]
    full_text = "\n".join(text)
    return full_text    

def get_article_content(query, search_google=False):
    urls = re.findall("(?P<url>https?://[^\\s]+)", query)
    if len(urls) == 0 and search_google:
        print("Searching google")
        urls = parse_google_search_results(query)
    elif len(urls) == 0 and not search_google:
        print("No URLs found")
        return ""
    pages_content = [parse_webpage_content(url) for url in urls]
    all_pages_content = "\n".join(pages_content)

    return all_pages_content

