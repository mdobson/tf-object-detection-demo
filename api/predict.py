from flask import Flask, request, Response, json, send_file
from werkzeug import secure_filename
from PIL import Image
import os
from io import BytesIO

import grpc
import numpy as np
from tensorflow.python.saved_model import signature_constants
from object_detection_demo.object_detection_compiled_protos import prediction_service_pb2
from object_detection_demo.object_detection_compiled_protos import prediction_service_pb2_grpc
from object_detection_demo.object_detection_compiled_protos import predict_pb2
from object_detection_demo.object_detection_compiled_protos import predict_pb2_grpc
from object_detection_demo.utils import label_map_util
from object_detection_demo.utils import visualization_utils as vis_util

import tensorflow as tf

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

app = Flask(__name__)
@app.route("/")
def hello():
	return "hello world"

@app.route('/app', methods=['GET'])
def serve_miniapp():
	content = open('index.html').read()	
	return Response(content, mimetype="text/html")

@app.route("/predict/image", methods = ["POST"])
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

	boxes = result.outputs['detection_boxes'].float_val
	classes = result.outputs['detection_classes'].float_val
	scores = result.outputs['detection_scores'].float_val
	
	if request.headers['accept'] == 'application/json':
		scored_result = [] 
		for i in range(0, len(scores)):
			cls = classes[i]
			score = scores[i]
			class_name = category_index[cls]['name']
			scored_result.append({ "class": class_name, "score": score})
	
		response_data = json.dumps(scored_result)
		resp = Response(response_data, status=200, mimetype='application/json')
		return resp

	else:
		image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			np.reshape(boxes, [100,4]),
			np.squeeze(classes).astype(np.int32),
			np.squeeze(scores),
			category_index,
			use_normalized_coordinates=True,
			line_thickness=2,
			min_score_thresh=.05)
		new_i = Image.fromarray(image_vis)	
		byte_io = BytesIO()
		new_i.save(byte_io, 'JPEG')
		byte_io.seek(0)

		return send_file(byte_io, as_attachment=True, attachment_filename="img.jpg", mimetype='image/jpg')

	
