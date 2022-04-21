import flask
import json
import os
from flask_cors import CORS
from bs4 import BeautifulSoup
import pymongo
from db_manager import DBManager

# Create the application.
app = flask.Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "Hello World!"


db = DBManager()


@app.route('/runs')
def runs():
    print('Getting runs!')
    cursor = db.get_runs()
    runs = []
    for doc in cursor:
        runs.append(doc)

    return flask.jsonify(runs)


@app.route('/runs/list')
def runs_list():
    print('getting list...')
    cursor = db.get_runs_list()
    runs = []
    for doc in cursor:
        id = doc.pop('_id')
        runs.append({'id': str(id), 'symbol': doc['meta']['symbol']})
    return flask.jsonify(runs)


@app.route('/run/')
def run():
    run_id = flask.request.args.get('id')

    print('Getting run:', run_id)
    cursor = db.get_run(run_id)
    run = []
    for doc in cursor:
        run.append(doc)

    return flask.jsonify(run[0])


if __name__ == '__main__':
    app.run(debug=True, port=8888)
