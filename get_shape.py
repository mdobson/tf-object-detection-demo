#Get the shape of the input tensor. See if we can feed it images without
#resizing.
import tensorflow as tf
import sys

from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.utils import build_tensor_info

model_dir = './ssd_mobilenet_v1_coco_2017_11_17/saved_model'
input_tensor = 'image_tensor'

#retrieve tensor data structures
with tf.Session() as sess:
  tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_dir)
  model_input_tensor = sess.graph.get_tensor_by_name(input_tensor+':0')
  print(model_input_tensor.get_shape())