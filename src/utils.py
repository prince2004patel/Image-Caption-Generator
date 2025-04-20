import json
import pickle

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)