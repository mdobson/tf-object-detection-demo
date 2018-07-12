from flask import Flask, json, Response, send_file, request
from flask_cors import CORS
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import os
app = Flask(__name__)
CORS(app)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


@app.route('/predict', methods = ['POST', 'GET'])
def hello_world():
	data = json.dumps([{"class": "dog", "score": 0.9406908750534058}, {"class": "dog", "score": 0.9345025420188904}, {"class": "dog", "score": 0.23088233172893524}, {"class": "dog", "score": 0.22518928349018097}])
	resp = Response(data, status=200, mimetype='application/json')
	return resp

@app.route('/app', methods=['GET'])
def serve_miniapp():
	content = open('index.html').read()	
	return Response(content, mimetype="text/html")

@app.route('/predict/image', methods = ['POST'])
def serve_image():
	img = request.files["images"]
	i = Image.open(img.stream)
	arr = load_image_into_numpy_array(i)
	new_i = Image.fromarray(arr)	
	byte_io = BytesIO()
	new_i.save(byte_io, 'JPEG')
	byte_io.seek(0)

	return send_file(byte_io, as_attachment=True, attachment_filename="img.jpg", mimetype='image/jpg')
