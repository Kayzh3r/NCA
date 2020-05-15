from src.DBManager import DBManager
from src.LibrivoxScraper import LibrivoxScraper, Track
import audioread
import os
import zipfile
import copy
import re


class AudioBooksManager:
    def __init__(self, db=DBManager(), driver=None):
        self.db = db
        self.__languages = ['spanish']
        self.__librivoxScraper = LibrivoxScraper(driver)
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
                trackList = []
                if book.url == '':
                    continue
                filename = os.path.join(dstPath, os.path.basename(book.url))
                book.dummy = os.path.splitext(os.path.basename(book.url))[0]
                if not os.path.isfile(filename):
                    downloadNow = True
                    self.__librivoxScraper.downloadFile(url=book.url, filename=filename, downloadLib='requests')
                bookFolder = os.path.join(dstPath, book.dummy)
                if not os.path.exists(bookFolder):
                    os.mkdir(bookFolder)
                with zipfile.ZipFile(filename, 'r') as zipId:
                    fileList = zipId.namelist()
                    for file in fileList:
                        trackPath = os.path.join(bookFolder,file.title())
                        if not os.path.isfile(trackPath):
                            zipId.extract(file.title(), trackPath)
                if downloadNow:
                    if self.db.audioBookExist(book.dummy):
                        self.db.audioBookUpdateStatusByName(book.dummy, 'DELETED')
                if not (not downloadNow and self.db.audioBookExist(book.dummy)):
                    for trackFile in os.listdir(bookFolder):
                        track = copy.deepcopy(book)
                        with audioread.audio_open(trackFile) as fId:
                            track.channels   = fId.channels
                            track.sampleRate = fId.samplerate
                            track.duration   = fId.duration
                        trackList.append(track)
                        self.db.audioBookCreate(trackList)
                if sizeMBDownloaded > sizeMB:
                    break
            if sizeMBDownloaded > sizeMB:
                break


if __name__ == '__main__':
    audioBooksManager = AudioBooksManager(driver=r"C:\Program Files (x86)\Google\ChromeDriver\chromedriver.exe")
    audioBooksManager.downloadData()
