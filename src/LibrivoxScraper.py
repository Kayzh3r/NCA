import time
import random
import os
from selenium import webdriver
from bs4 import BeautifulSoup


class LibrivoxScraper():
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

    def setDriver(self, webDriver=''):
        self.__driverPath = webDriver
        if not os.path.exists(self.__driverPath):
            print('Driver does not exist')
            self.__wrongInit = True
        else:
            self.__wrongInit = False
            self.__browser = webdriver.Chrome(self.__driverPath)

    def getBooksByLanguage(self, language):
        if self.__wrongInit:
            print('Driver does not exist')
            return None
        if language not in self.__languages.keys():
            print('Unknown requested language')
            print('Known languages are:')
            print(self.__languages.keys())
            return None
        availablePage = True
        page = 1
        while availablePage:
            url = self.__mainURL + \
                  self.__primaryKey + str(self.__languages[language]) + \
                  self.__searchCategory + self.__categories['language'] + \
                  self.__searchPage + str(page) + \
                  self.__searchForm
            self.__browser.get(url)
            time.sleep(random.uniform(0.3, 0.7))
            soup = BeautifulSoup(self.__browser.page_source, "html.parser")
            results = soup.findAll('li', {'class': 'catalog-result'})
            #for results in results:

            page += 1
        self.__browser.close()


if __name__ == '__main__':
    myScraper = LibrivoxScraper(r"C:\Program Files (x86)\Google\ChromeDriver\chromedriver.exe")
    myScraper.getBooksByLanguage('spanish')
