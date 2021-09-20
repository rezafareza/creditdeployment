from flask import Flask, jsonify, json, request
from logging import debug

import pickle
from predict import makeprediction

app = Flask(__name__)
users = []


@app.route('/', methods=['GET'])
def hello():
    return "HELLO WORLD"



@app.route("/pred", methods=['POST'])
def make_predictions():
    ''' DO Somehting
    '''
    if request.method == 'POST':
        data = request.get_json()
        # print(data)
        result = makeprediction(data)
        result = {
                'model': 'Model-LR_8_features',
                'version': '1.0.0',
                'score_proba': result['data'][0]['pred_proba'],
                'prediction': result['data'][0]['prediction']
            }	
        print(result)
        return result


if __name__ == '__main__':
    app.run(port=5000, debug=False)
