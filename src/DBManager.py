import sqlite3
import os
from socket import gethostname
from getpass import getuser


class DBManager:
    def __init__(self):
        self.isOpen = False

        ''' Private attributes'''
        self.__name = './model.db'
        self.__exists = False
        self.__conn = None
        self.__cursor = None

        self.createDB()

    def __connect(self):
        self.__conn = sqlite3.connect(self.__name)
        self.__cursor = self.__conn.cursor()
        self.isOpen = True

    def __close(self):
        self.__conn.close()
        self.__cursor = None
        self.__conn = None
        self.isOpen = False

    def createDB(self):
        try:
            if os.path.exists(self.__name):
                self.__exists = True
            else:
                self.__connect()
                query = "CREATE TABLE IF NOT EXISTS model_checkpoint (".__add__(
                        "id integer PRIMARY KEY,").__add__(
                        "datetime int NOT NULL,").__add__(
                        "user text NOT NULL,").__add__(
                        "machine text NOT NULL,").__add__(
                        "model_name text NOT NULL,").__add__(
                        "model_version text NOT NULL,").__add__(
                        "checkpoint_status text NOT NULL,").__add__(
                        "checkpoint_path text NOT NULL)")
                self.__cursor.execute(query)
                query = "CREATE TABLE IF NOT EXISTS model_info (".__add__(
                        "name text NOT NULL,").__add__(
                        "version text NOT NULL,").__add__(
                        "status text NOT NULL,").__add__(
                        "path text NOT NULL,").__add__(
                        "CONSTRAINT pk_model_info PRIMARY KEY (name, version))")
                self.__cursor.execute(query)
                query = "CREATE TABLE IF NOT EXISTS noise_files (".__add__(
                        "id integer PRIMARY KEY,").__add__(
                        "name text NOT NULL,").__add__(
                        "url text NOT NULL,").__add__(
                        "path text NOT NULL,").__add__(
                        "channels int NOT NULL,").__add__(
                        "sample_rate int NOT NULL,").__add__(
                        "duration real NOT NULL,").__add__(
                        "status text NOT NULL,").__add__(
                        "insert_datetime int NOT NULL)")
                self.__cursor.execute(query)
                query = "CREATE TABLE IF NOT EXISTS audio_books_tracks (".__add__(
                        "id integer PRIMARY KEY,").__add__(
                        "book_dummy_name text NOT NULL,").__add__(
                        "book_name text NOT NULL,").__add__(
                        "book_author text NOT NULL,").__add__(
                        "book_url text NOT NULL,").__add__(
                        "book_language text NOT NULL,").__add__(
                        "book_path text NOT NULL,").__add__(
                        "book_n_tracks int NOT NULL,").__add__(
                        "track_name text NOT NULL,").__add__(
                        "track_path text NOT NULL,").__add__(
                        "track_channels int NOT NULL,").__add__(
                        "track_sample_rate int NOT NULL,").__add__(
                        "track_duration real NOT NULL,").__add__(
                        "track_status text NOT NULL,").__add__(
                        "track_insert_datetime int NOT NULL)")
                self.__cursor.execute(query)
                query = "CREATE TABLE IF NOT EXISTS training_track (".__add__(
                    "id integer PRIMARY KEY,").__add__(
                    "model_name text NOT NULL,").__add__(
                    "model_version text NOT NULL,").__add__(
                    "start_checkpoint_id integer NOT NULL,").__add__(
                    "end_checkpoint_id integer NOT NULL,").__add__(
                    "audio_book_track_id integer NOT NULL,").__add__(
                    "noise_id integer NOT NULL,").__add__(
                    "epoch integer NOT NULL,").__add__(
                    "status text NOT NULL,").__add__(
                    "insert_datetime int NOT NULL)")
                self.__cursor.execute(query)
                self.__conn.commit()
                self.__close()
        except Exception:
            self.__close()

    def noiseExist(self, name):
        try:
            self.__connect()
            self.__cursor.execute(
                "SELECT name " +
                "FROM noise_files " +
                "WHERE name = '" + name + "'"
                )
            cursorVal = self.__cursor.fetchall()
            if not cursorVal:
                retVal = False
            else:
                retVal = True
            self.__close()
            return retVal
        except Exception as error:
            self.__close()

    def noiseCreate(self, name, url, path, channels, sampleRate, duration):
        try:
            self.__connect()
            self.__cursor.execute(
                "SELECT COALESCE(MAX(id) + 1,1) FROM noise_files"
            )
            newId = self.__cursor.fetchall()
            query = "INSERT INTO noise_files ".__add__(
                    "(id, name, url, path, channels, sample_rate, duration, status, insert_datetime) ").__add__(
                    "VALUES (?,?,?,?,?,?,?,?, datetime('now'))")
            self.__cursor.execute(query, [int(newId[0][0]), name, url, path,
                                          int(channels), int(sampleRate), duration, "OK"])
            self.__conn.commit()
            self.__close()
        except Exception as error:
            self.__close()

    def noiseGetByName(self, name):
        try:
            self.__connect()
            self.__cursor.execute(
                "SELECT * FROM noise_files " +
                "WHERE name = '" + name + "' " +
                "ORDER BY insert_datetime DESC " +
                "LIMIT 1"
            )
            retQuery = self.__cursor.fetchall()
            self.__close()
            return retQuery
        except Exception as error:
            self.__close()
            return None

    def noiseUpdateStatusByName(self, name, status):
        try:
            self.__connect()
            query = "UPDATE noise_files ".__add__(
                    "SET status = '" + status + "' ").__add__(
                    "WHERE id = ( ").__add__(
                    "SELECT id ").__add__(
                    "FROM noise_files ").__add__(
                    "WHERE name = '" + name + "' ").__add__(
                    "ORDER BY insert_datetime DESC ").__add__(
                    "LIMIT 1").__add__(
                    ")")
            self.__cursor.execute(query)
            self.__conn.commit()
            self.__close()
        except Exception as error:
            self.__close()

    def modelExist(self, name):
        try:
            self.__connect()
            self.__cursor.execute(
                "SELECT name " +
                "FROM model_info " +
                "WHERE name = '" + name + "'"
                )
            cursorVal = self.__cursor.fetchall()
            if not cursorVal:
                retVal = False
            else:
                retVal = True
            self.__close()
            return retVal
        except Exception as error:
            self.__close()

    def modelCreate(self, name, version, pyFilePath, checkpointPath):
        try:
            self.__connect()
            query = "INSERT INTO model_info ".__add__(
                    "(name, version, status, path) ").__add__(
                    "VALUES (?,?,?,?)")
            self.__cursor.execute(query, [name, version, "CREATED", pyFilePath])
            self.__cursor.execute(
                    "SELECT COALESCE(MAX(id) + 1,1) FROM model_checkpoint"
            )
            newId = self.__cursor.fetchall()[0][0]
            query = "INSERT INTO model_checkpoint ".__add__(
                    "(id, datetime, user, machine, model_name, model_version, ").__add__(
                    "checkpoint_status, checkpoint_path) ").__add__(
                    "VALUES (?,datetime('now'),?,?,?,?,?,?)")
            self.__cursor.execute(query, [int(newId), getuser(),
                                          gethostname(), name, version, "CREATED", checkpointPath])
            self.__conn.commit()
            self.__close()
        except Exception as error:
            self.__close()

    def modelGetInfo(self, name, ver):
        try:
            self.__connect()
            query = "SELECT * ".__add__(
                    "FROM model_info ").__add__(
                    "WHERE name ='" + name + "'").__add__(
                    "AND version = '" + ver + "'")
            self.__cursor.execute(query)
            cursorVal = self.__cursor.fetchall()
            if not cursorVal:
                retVal = None
            else:
                retVal = {'name'    : cursorVal[0][0],
                          'version' : cursorVal[0][1],
                          'status'  : cursorVal[0][2],
                          'path'    : cursorVal[0][3]}
                query = "SELECT * ".__add__(
                        "FROM training_execution ").__add__(
                        "WHERE name ='" + name + "' ").__add__(
                        "AND version ='" + ver + "' ").__add__(
                        "ORDER BY insert_datetime DESC ").__add__(
                        "LIMIT 1)")
                self.__cursor.execute(query)
                cursorVal = self.__cursor.fetchall()
                retVal['checkpoint_status'] = cursorVal[0][6]
                retVal['checkpoint_path'] = cursorVal[0][7]
            self.__close()
            return retVal
        except Exception as error:
            self.__close()

    def modelTrainNext(self, name, ver):
        try:
            self.__connect()
            query = "SELECT * ".__add__(
                "FROM  training_track ").__add__(
                "WHERE model_name ='" + name + "' ").__add__(
                "AND model_version = '" + ver + "' ").__add__(
                "AND status = 'NO TRAINED' ").__add__(
                "ORDER BY insert_datetime ASC ").__add__(
                "LIMIT 1"
            )
            self.__cursor.execute(query)
            cursorVal = self.__cursor.fetchall()
            self.__close()
            return cursorVal
        except Exception as error:
            self.__close()

    def audioBookExist(self, name):
        try:
            self.__connect()
            self.__cursor.execute(
                "SELECT book_dummy_name " +
                "FROM audio_books_tracks " +
                "WHERE book_name = '" + name + "'"
            )
            cursorVal = self.__cursor.fetchall()
            if not cursorVal:
                retVal = False
            else:
                retVal = True
            self.__close()
            return retVal
        except Exception as error:
            self.__close()

    def audioBookCreate(self, trackList):
        try:
            self.__connect()
            for track in trackList:
                self.__cursor.execute(
                    "SELECT COALESCE(MAX(id) + 1,1) FROM audio_books_tracks"
                )
                newId = self.__cursor.fetchall()
                query = "INSERT INTO audio_books_tracks ".__add__(
                    "(id, book_dummy_name, book_name, book_author, book_url, book_language, book_path, ").__add__(
                    "book_n_tracks, track_name, track_path, track_channels, track_sample_rate, ").__add__(
                    "track_duration, track_status, track_insert_datetime) ").__add__(
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, datetime('now'))")
                self.__cursor.execute(query, [int(newId[0][0]), track.dummy, track.title, track.authorName, track.url,
                                              track.language, track.zip, track.nTracks, track.name, track.path, track.channels,
                                              track.sampleRate, track.duration, "OK"])
            self.__conn.commit()
            self.__close()
        except Exception as error:
            self.__close()

    def audioBookGetByName(self, name):
        try:
            self.__connect()
            self.__cursor.execute(
                "SELECT book_n_tracks FROM audio_books_tracks " +
                "WHERE book_name = '" + name + "' " +
                "ORDER BY track_insert_datetime DESC " +
                "LIMIT 1"
            )
            nTracks = self.__cursor.fetchall()
            self.__cursor.execute(
                "SELECT * FROM audio_books_tracks " +
                "WHERE book_name = '" + name + "' " +
                "ORDER BY insert_datetime DESC " +
                "LIMIT " + str(int(nTracks[0][0]))
            )
            retQuery = self.__cursor.fetchall()
            self.__close()
            return retQuery
        except Exception as error:
            self.__close()
            return None

    def audioBookUpdateStatusByName(self, name, status):
        try:
            self.__connect()
            self.__cursor.execute(
                "SELECT book_n_tracks FROM audio_books_tracks " +
                "WHERE book_name = '" + name + "' " +
                "ORDER BY track_insert_datetime DESC " +
                "LIMIT 1"
            )
            nTracks = self.__cursor.fetchall()
            query = "UPDATE audio_books_tracks ".__add__(
                "SET status = '" + status + "' ").__add__(
                "WHERE book_name = " + "'" + name + "' ").__add__(
                "ORDER BY insert_datetime DESC ").__add__(
                "LIMIT " + str(int(nTracks[0][0]))
            )
            self.__cursor.execute(query)
            self.__conn.commit()
            self.__close()
        except Exception as error:
            self.__close()


if __name__ == '__main__':
    bdManager = DBManager()
    bdManager.createDB()
