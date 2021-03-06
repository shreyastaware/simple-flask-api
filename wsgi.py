from flask import Flask, request, jsonify
import pickle, json
import numpy as np

global __data_columns
global __model

app = Flask(__name__)

with open("/app/columns.json", "r") as f:
    __data_columns = json.load(f)['data_columns']

with open("/app/model.pickle", "rb") as f:
    __model = pickle.load(f)

if __name__ == "__main__":
    app.run()

@app.route('/housing-prices', methods = ['GET', 'POST'])
def predict_alphanumeric_text():

    data = request.form

    lis = []

    for l in __data_columns:
        for keys in data:
            if keys.lower() == l:
                lis.append(data[keys])

    s = np.array(lis)

    s = s.reshape(1, s.shape[0])

    label = __model.predict(s)[0]
    
    response = jsonify({
		"probability": 0.9, 
		"request_id": 4, 
		"result": label, 
		"status": 1 
	})

    return response
