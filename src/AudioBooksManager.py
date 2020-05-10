from src.DBManager import DBManager
from src.LibrivoxScraper import LibrivoxScraper, Book
import audioread
import os
import requests
import re

class AudioBooksManager:
    def __init__(self, db=DBManager()):
        self.db = db
        self.__languages = ['spanish']
        self.__librivoxScraper = LibrivoxScraper(r"C:\Program Files (x86)\Google\ChromeDriver\chromedriver.exe")
        self.__books = {}

    def __getBooks(self, language):
        self.__books[language] = self.__librivoxScraper.getBooksByLanguage(language)

    def downloadData(self, dstPath='./downloads', sizeMB=0):
        downloadNow = False
        sizeMBDownloaded = 0
        if not os.path.exists(dstPath):
            os.mkdir(dstPath)
        for language in self.__languages:
            self.__getBooks(language)
            for book in self.__books[language]:
                if book.url == '':
                    continue
                filename = os.path.join(dstPath, os.path.basename(book.url))
                if not os.path.isfile(filename):
                    downloadNow = True
                    request = requests.get(book.url)
                    with open(filename, 'wb') as fId:
                        fId.write(request.content)

                '''if downloadNow:
                    if self.db.noiseExist(key):
                        self.db.noiseUpdateStatusByName(key, 'DELETED')
                if not (not downloadNow and self.db.noiseExist(key)):
                    with audioread.audio_open(filename) as fId:
                        self.db.noiseCreate(key, self.resources[key],
                                            filename, fId.channels,
                                            fId.samplerate, fId.duration)'''
                if sizeMBDownloaded > sizeMB:
                    break
            if sizeMBDownloaded > sizeMB:
                break


if __name__ == '__main__':
    audioBooksManager = AudioBooksManager()
    audioBooksManager.downloadData()
