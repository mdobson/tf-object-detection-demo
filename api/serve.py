from flask import Flask, json, Response
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods = ['POST', 'GET'])
def hello_world():
	data = json.dumps([{"class": "dog", "score": 0.9406908750534058}, {"class": "dog", "score": 0.9345025420188904}, {"class": "dog", "score": 0.23088233172893524}, {"class": "dog", "score": 0.22518928349018097}])
	resp = Response(data, status=200, mimetype='application/json')
	return resp

