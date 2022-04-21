from utils.singleton import Singleton
import json
import datetime
from constants import constants
from server.db_manager import DBManager
import socket

@Singleton
class Documentor:
    data = {
        "losses": [],
        "meta": constants,
        "performance": [],
        "ran_by": socket.gethostname()
    }
    lossCount = 0

    def __init__(self):
        print("Documentor initialized")

    def add_loss(self, value):
        # Only store every 50th loss, to reduce the size of the json file
        if (self.lossCount % 50 == 0):
            self.data["losses"].append(value)
        self.lossCount += 1
    
    def add_performance(self, performance):
        self.data["performance"].append(performance)

    def write(self, key, value):
        self.data[key] = value

    def read(self):
        print(self.data)

    def save(self):
        now = datetime.datetime.now()
        # trim self.data['losses'] to only the first 1000 values
        self.data['losses'] = self.data['losses'][:1000]
        self.data['meta']['date_ran'] = now
        db = DBManager()
        db.add_run(self.data)