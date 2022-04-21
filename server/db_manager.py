import pymongo
from bson.objectid import ObjectId
class DBManager:
    url = "mongodb+srv://admin:admin@cluster0.km4a7.mongodb.net/Ample?retryWrites=true&w=majority"

    def __init__(self):
        self.client = pymongo.MongoClient(self.url)
        self.db = self.client.Ample
        self.runs = self.db.runs
        self.env = self.db.env

    def add_run(self, run):
        self.runs.insert_one(run)

    def get_runs(self):
        return self.runs.find({}, {"_id": 0})

    def get_run(self, run_id):
        return self.runs.find({"_id": ObjectId(run_id)}, {"_id": 0})

    def get_runs_list(self):
        return self.runs.find({}, {"_id": 1, "meta.symbol": 1})

    def get_env(self, symbol):
        return self.env.find({"symbol": symbol}, {"_id": 0})

    def get_env_list(self):
        return self.env.find({}, {"_id": 0})

    def update_env(self, symbol, data):
        self.env.update_one({"symbol": symbol}, {
                            "$set": {"data": data}}, upsert=True)

    def add_env(self, symbol, data):
        self.env.insert_one({"symbol": symbol, "data": data})
