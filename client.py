from PIL import Image
import numpy as np

import grpc
from tensorflow.python.saved_model import signature_constants
import prediction_service_pb2
import prediction_service_pb2_grpc
import predict_pb2
import predict_pb2_grpc
import tensorflow as tf

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


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
print(result)

 

