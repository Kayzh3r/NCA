import grequests
import urllib.request
from bs4 import BeautifulSoup

class LibrivoxScraper:
    def __init__(self):
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

    def getBooksByLanguage(self, language):
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
            response = grequests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.findAll('li', {'class': 'catalog-result'})
            page += 1


if __name__ == '__main__':
    myScraper = LibrivoxScraper()
    myScraper.getBooksByLanguage('spanish')