import json


def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w+') as file:
        json.dump(data, file)


def load_json(file_path):
    assert file_path.split('.')[-1] == 'json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data