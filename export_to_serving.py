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

#input and output nodes
input_tensor = 'image_tensor'
output_tensors = ['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections']

serving_model_path = './serving_model/ssd_mobilenet_v1_coco_2017_11_17/1'
model_filename = './ssd_mobilenet_v1_coco_2017_11_17/saved_model/saved_model.pb'
model_dir = './ssd_mobilenet_v1_coco_2017_11_17/saved_model'

#retrieve tensor data structures
with tf.Session() as sess:
	tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_dir)
	
	model_input_tensor = sess.graph.get_tensor_by_name(input_tensor+':0')
	detection_boxes_output_tensor = sess.graph.get_tensor_by_name(output_tensors[0]+':0')
	detection_scores_output_tensor = sess.graph.get_tensor_by_name(output_tensors[1]+':0')
	detection_classes_output_tensor = sess.graph.get_tensor_by_name(output_tensors[2]+':0')
	num_detections_output_tensor = sess.graph.get_tensor_by_name(output_tensors[3]+':0')

	model_input = build_tensor_info(model_input_tensor)

	detection_boxes = build_tensor_info(detection_boxes_output_tensor)
	detection_scores = build_tensor_info(detection_scores_output_tensor)
	detection_classes = build_tensor_info(detection_classes_output_tensor)
	num_detections = build_tensor_info(num_detections_output_tensor)

	outputs = {
		output_tensors[0]: detection_boxes,
		output_tensors[1]: detection_scores,
		output_tensors[2]: detection_classes,
		output_tensors[3]: num_detections
	}


	signature_definition = signature_def_utils.build_signature_def(
		inputs={input_tensor: model_input},
		outputs=outputs,
		method_name=signature_constants.PREDICT_METHOD_NAME)

	builder = saved_model_builder.SavedModelBuilder(serving_model_path)
	
	builder.add_meta_graph_and_variables(
		sess, [tag_constants.SERVING],
		signature_def_map={
			signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
				signature_definition
		})
			

	builder.save()


