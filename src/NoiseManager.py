from __future__ import unicode_literals
import youtube_dl
import os


class NoiseManager():
    def __init__(self):
        self.resources = {
            'airDryer': 'https://www.youtube.com/watch?v=PNAGqh2h3AA',
            'serverRoom': 'https://www.youtube.com/watch?v=gLLvXi1Usrc',
            'coffeeShop': 'https://www.youtube.com/watch?v=BOdLmxy06H0',
            'crowd': 'https://www.youtube.com/watch?v=IKB3Qiglyro',
            'peopleTalking': 'https://www.youtube.com/watch?v=PHBJNN-M_Mo',
            'cityRain': 'https://www.youtube.com/watch?v=eZe4Q_58UTU',
            'city1': 'https://www.youtube.com/watch?v=cDWZkXjDYsc',
            'city2': 'https://www.youtube.com/watch?v=YF3pj_3mdMc',
            'city3': 'https://www.youtube.com/watch?v=8s5H76F3SIs',
            'city4': 'https://www.youtube.com/watch?v=Vg1mpD1BICI'
        }
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl' : '',
        }

    def downloadData (self, dstPath='./downloads'):
        if not os.path.exists(dstPath):
            os.mkdir(dstPath)
        for key in self.resources:
            filename = os.path.join(dstPath, key)
            if not os.path.isfile(filename):
                self.ydl_opts['outtmpl'] = filename
                with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                    success = ydl.download([self.resources[key]])


if __name__ == '__main__':
    noiseManager = NoiseManager()
    noiseManager.dataDownload()
