#!/usr/bin/env python






#### THANE/JUSTIN: THIS IS THE FILE THAT WILL EVOLVE INTO THE PERSON-FOLLOW NODE #####






# # This script detects tennis balls in a video feed.
# Be sure to follow the [installation instructions]
# (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
# before you start.
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation
# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# # Detection
# Choose video feed to be a video file (e.g. 'path/to/file') or a camera input (e.g. 0 or 1)
# cap = cv2.VideoCapture('/home/seth/Videos/vid3.mp4')
# cap = cv2.VideoCapture('/home/seth/Videos/urc_autonomy/A4.MOV')
cap = cv2.VideoCapture(0)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    while cap.isOpened():
        ret, frame = cap.read()
        image_np = frame # use this line for any video feed that isn't the ZED
        # image_np = frame[0:480, 0:640] # for using only the left camera feed if using ZED as input

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #   image_np,
        #   np.squeeze(boxes),
        #   np.squeeze(classes).astype(np.int32),
        #   np.squeeze(scores),
        #   category_index,
        #   use_normalized_coordinates=True,
        #   line_thickness=8)
        box = boxes[0][0]
        im_h, im_w, _ = image_np.shape
        corners = (int(box[1]*im_w), int(box[0]*im_h), int(box[3]*im_w), int(box[2]*im_h))
        if scores[0][0] > .50:
            pass
            # Instead of fancy probability box, show simple rectangle
            # cv2.rectangle(image_np, (corners[0], corners[1]), (corners[2], corners[3]), (255,0,0), 2)
        else:
            corners = (-1, -1, -1, -1)
        print(str(corners) + ', {:1.2f}'.format(scores[0][0]))
        # cv2.imshow('object detection 3', cv2.resize(image_np, (640, 480)))
        k = 0
        for i in range(classes.size):
            if classes[0][i] == 1:
                k = i

        
            
        cv2.imshow('object detection 2', image_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

    cv2.destroyAllWindows()
    cap.release()
