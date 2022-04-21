from utils.singleton import Singleton


@Singleton
class ModelManager:
    def __init__(self):
        self.models = {}

    def load_model(self, path):
        pass

    def get_model(self, symbol):
        pass

    def receive_data(self, symbol, data):
        '''
        Recieves data from cron_handler and passes it to the respective model
        '''
        pass
