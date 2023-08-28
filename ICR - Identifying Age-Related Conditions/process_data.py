import json

from script.preprocess import preprocess_data

if __name__== '__main__':
    
    with open('config.json', 'r') as file:
        config = json.load(file)

    preprocess_data(config)