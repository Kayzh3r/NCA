from tensorflow import keras
from src.DBManager import DBManager
from src.NoiseManager import NoiseManager

class NCA:
    def __init__(self):
        self.db = DBManager()
        self.noise = NoiseManager(self.db)

        # Calling initialize method
        self.initialize()

    def initialize(self):


    def __saveModel(self):
        self.model.save()

    def __loadModel(self):
        model = keras.models.load_model('path_to_my_model.h5')



if __name__ == "__main__":
