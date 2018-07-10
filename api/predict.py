from flask import Flask, request, Response, json
from werkzeug import secure_filename
from PIL import Image
import os

import grpc
import numpy as np
from tensorflow.python.saved_model import signature_constants
from object_detection_demo.object_detection_compiled_protos import prediction_service_pb2
from object_detection_demo.object_detection_compiled_protos import prediction_service_pb2_grpc
from object_detection_demo.object_detection_compiled_protos import predict_pb2
from object_detection_demo.object_detection_compiled_protos import predict_pb2_grpc
from object_detection_demo.utils import label_map_util

import tensorflow as tf

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

app = Flask(__name__)
@app.route("/")
def hello():
	return "hello world"

@app.route("/predict", methods = ["POST"])
def predict():
	img = request.files["images"]
	image = Image.open(img.stream)
	PATH_TO_LABELS = os.path.join('../data', 'mscoco_label_map.pbtxt')

	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	channel = grpc.beta.implementations.insecure_channel('172.83.15.19', 9000)
	stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

	rpc_request = predict_pb2.PredictRequest()

	rpc_request.model_spec.name = 'obj_det'

	image_np = load_image_into_numpy_array(image)
	image_np_expanded = np.expand_dims(image_np, axis=0)

	rpc_request.model_spec.signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
	rpc_request.inputs['image_tensor'].CopyFrom(tf.contrib.util.make_tensor_proto(image_np_expanded))
	result = stub.Predict(rpc_request, 10.0)

	classes = result.outputs['detection_classes'].float_val
	scores = result.outputs['detection_scores'].float_val

	scored_result = [] 
	for i in range(0,4):
		cls = classes[i];
		score = scores[i];
		class_name = category_index[cls]['name']
		scored_result.append({class_name: score})
	
	response_data = json.dumps(scored_result)
	resp = Response(response_data, status=200, mimetype='application/json')
	return resp


