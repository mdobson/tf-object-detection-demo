from PIL import Image
import numpy as np

import os

from  utils import label_map_util

import scipy


import grpc
from tensorflow.python.saved_model import signature_constants
from object_detection_compiled_protos import prediction_service_pb2
from object_detection_compiled_protos import prediction_service_pb2_grpc
from object_detection_compiled_protos import predict_pb2
from object_detection_compiled_protos import predict_pb2_grpc
import tensorflow as tf

from utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_name', 'image1.jpg', """Image to send for prediction""")

channel = grpc.beta.implementations.insecure_channel('172.83.15.19', 9000)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()

request.model_spec.name = 'obj_det'

image = Image.open(FLAGS.image_name)
image_np = load_image_into_numpy_array(image)
image_np_expanded = np.expand_dims(image_np, axis=0)

request.model_spec.signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
request.inputs['image_tensor'].CopyFrom(tf.contrib.util.make_tensor_proto(image_np_expanded))
result = stub.Predict(request, 10.0)

boxes = result.outputs['detection_boxes'].float_val
classes = result.outputs['detection_classes'].float_val
scores = result.outputs['detection_scores'].float_val
image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
	image_np,
	np.reshape(boxes, [100,4]),
	np.squeeze(classes).astype(np.int32),
	np.squeeze(scores),
	category_index,
	use_normalized_coordinates=True,
	line_thickness=3)

scipy.misc.imsave('analyzed.jpg'.format(FLAGS.image_name), image_vis)
 

