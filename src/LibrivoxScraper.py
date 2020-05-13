import time
import random
import os
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm


class Book:
    def __init__(self, dataInfo, downloadInfo):
        self.title      = dataInfo['title']
        self.authorName = dataInfo['author']['name']
        self.authorUrl  = dataInfo['author']['url']
        self.state      = dataInfo['metadata']['state']
        self.type       = dataInfo['metadata']['type']
        self.language   = dataInfo['metadata']['language']
        if downloadInfo:
            self.url  = downloadInfo['url']
            self.size = downloadInfo['size']
        else:
            self.url  = ''
            self.size = ''


class LibrivoxScraper:
    def __init__(self, webDriver=''):
        self.__mainURL     = r"https://librivox.org/search?"
        self.__primaryKey  = r"primary_key="
        self.__searchCategory = "&search_category="
        self.__searchPage  = "&search_page="
        self.__searchForm  = "&search_form=get_results"
        self.__languages = {'english' : 1,
                            'french'  : 2,
                            'german'  : 3,
                            'italian' : 4,
                            'spanish' : 5}
        self.__categories = {'author'   : 'author',
                             'title'    : 'title',
                             'language' : 'language'}
        self.__wrongInit = None
        self.__driverPath = None
        self.__browser = None
        self.setDriver(webDriver)
        self.__books = []

    def setDriver(self, webDriver=''):
        self.__driverPath = webDriver
        if not os.path.exists(self.__driverPath):
            print('Driver does not exist')
            self.__wrongInit = True
        else:
            self.__wrongInit = False

    def __parseDownload(self, result):
        downloadBtn = result.findAll('div', {'class': 'download-btn'})
        if len(downloadBtn) != 1:
            return None
        else:
            spanNode = downloadBtn[0].findAll('span')
            if len(spanNode) != 1:
                return None
            downloadSize = spanNode[0].contents[0]
            downloadUrl = downloadBtn[0].contents[1]['href']
            downloadInfo = {'url'  : downloadUrl,
                            'size' : downloadSize}
            return downloadInfo

    def __parseData(self, result):
        data = result.findAll('div', {'class': 'result-data'})
        bookTitle = data[0].findAll('h3')[0].findAll('a')[0].contents[0]
        bookAuthor = data[0].findAll('p', {'class': 'book-author'})
        if len(bookAuthor[0]) == 1:
            authorName = bookAuthor[0].contents[0].strip()
            authorUrl = ''
        else:
            authorName = bookAuthor[0].contents[1].contents[0].strip()
            authorUrl = bookAuthor[0].contents[1]['href']
        bookMeta = data[0].findAll('p', {'class': 'book-meta'})
        values = [x.strip() for x in bookMeta[0].contents[0].strip().split('|')]
        authorDict = {'name' : authorName,
                      'url'  : authorUrl}
        metadataDict = {'state'    : values[0],
                        'type'     : values[1],
                        'language' : values[2]}
        dataInfo = {'title'    : bookTitle,
                    'author'   : authorDict,
                    'metadata' : metadataDict}
        return dataInfo

    def getBooksByLanguage(self, language):
        if self.__wrongInit:
            print('Driver does not exist')
            return None
        self.__browser = webdriver.Chrome(self.__driverPath)
        if language not in self.__languages.keys():
            print('Unknown requested language')
            print('Known languages are:')
            print(self.__languages.keys())
            return None
        availablePage = True
        page = 1
        increaseSleep = 0
        lastPage = 0
        while availablePage:
            url = self.__mainURL + \
                  self.__primaryKey + str(self.__languages[language]) + \
                  self.__searchCategory + self.__categories['language'] + \
                  self.__searchPage + str(page) + \
                  self.__searchForm
            self.__browser.get(url)
            time.sleep(increaseSleep + random.uniform(0.5, 1))
            soup = BeautifulSoup(self.__browser.page_source, "html.parser")
            results = soup.findAll('li', {'class': 'catalog-result'})
            lastPageNode = soup.findAll('a', {'class': 'page-number last'})
            if len(lastPageNode) == 0:
                increaseSleep += 1
                continue
            else:
                increaseSleep = 0
            if lastPage == 0:
                lastPage = int(lastPageNode[0]['data-page_number'])
            for result in results:
                downloadInfo = self.__parseDownload(result)
                dataInfo = self.__parseData(result)
                if dataInfo:
                    self.__books.append(Book(dataInfo, downloadInfo))
            if page == lastPage:
                availablePage = False
            page += 1
        self.__browser.close()
        return self.__books

    def downloadFile(self, url='', filename='', downloadLib='requests'):
        if downloadLib == 'selenium':
            chrome_options = webdriver.ChromeOptions()
            prefs = {'download.default_directory': filename}
            chrome_options.add_experimental_option('prefs', prefs)
            if self.__wrongInit:
                print('Driver does not exist')
                return None
            self.__browser = webdriver.Chrome(self.__driverPath, chrome_options=chrome_options)
            self.__browser.get(url)
            self.__browser.close()
        else:
            response = requests.get(url, stream=True)
            with open(filename, "wb") as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)


if __name__ == '__main__':
    myScraper = LibrivoxScraper(r"C:\Program Files (x86)\Google\ChromeDriver\chromedriver.exe")
    books = myScraper.getBooksByLanguage('spanish')
    print(len(books))
