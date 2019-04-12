#!/usr/bin/env python
# # This script detects tennis balls in a video feed.
# Be sure to follow the [installation instructions]
# (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
# before you start.
from utils import visualization_utils as vis_util
from utils import label_map_util
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

sys.path.append("..")

class PersonFollow:
    def __init__(self):
        
        # Subscribers
        self.sub_img = rospy.Subscriber('/camera/color/image_rect_color', Image, self.imageCallback)
        self.sub_dep = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depthCallback)

        # Publishers
        self.pub_cmd = rospy.Publisher('')
        

        #member variables
        self.frameWidth = 640
        self.frameHeight = 480
        self.middlePixel = self.frameWidth/2.0

        self.bridge = CvBridge()
        self.msgread = {'image': False, 'depth': False}
        self.ready = False

        self.init()
        
    def imageCallback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as e:
            print(e)

    def depthCallback(self, msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as e:
            print(e)
            
    def check_ready(self):
        ready = True
        for key in self.msgread:
            if not self.msgread[key]:
                ready = False
        self.ready = ready
    
    def init(self):
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
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
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
                detection_boxes = detection_graph.get_tensor_by_name(
                    'detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def getDepth(self):
        #calculate position from depth matrix
        # return estimatedDepth
        pass

    def controller(self,upperLeftx, upperLefty, lowerRightx, lowerRighty):
        rectCenter = (lowerRightx - upperLeftx)/2
        orientationDiff = self.middlePixel - rectCenter
        translationDiff = self.desiredDistance - self.getDepth()

        turnGain = 5 # need tuned
        velocityGain = 5 # need tuned
        turnCommand = orientationDiff * turnGain
        velocityCommand = translationDiff * velocityGain
        
        # publish commands
        

    def run(self):

        while cap.isOpened():
            ret, frame = cap.read()
            image_np = frame  # use this line for any video feed that isn't the ZED
            # image_np = frame[0:480, 0:640] # for using only the left camera feed if using ZED as input

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores,
                    detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})





##################### LOGIC SECTION ##################################################            
            sz = 5
            boxes = np.squeeze(boxes)[:sz]
            classes = np.squeeze(classes).astype(np.int32)[:sz]
            scores = np.squeeze(scores)[:sz]
            # Visualization of the results of a detection.
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #   image_np,
            #   boxes,
            #   classes.astype(np.int32),
            #   scores,
            #   category_index,
            #   use_normalized_coordinates=True,
            #   line_thickness=8)
            k = -1
            for i in range(classes.size):
                if classes[i] == 1 and scores[i] > .40:
                    k = i
                    break

            if k == -1:
                print("No person detected")
                continue

            box = boxes[k]
            im_h, im_w, _ = image_np.shape
            corners = (int(box[1]*im_w), int(box[0]*im_h),
                       int(box[3]*im_w), int(box[2]*im_h))
            
            cv2.rectangle(image_np, (corners[0], corners[1]), (corners[2], corners[3]), (255,0,0), 4)
            
            # calculate 
            controller(corners[0], corners[1], corners[2], corners[3])

            print('Person detected with probability: {:1f}'.format(scores[k]*100))
            cv2.imshow('object detection', image_np)


if __name__ == '__main__':
    rospy.init_node('person_follow_node')
    node = PersonFollow()
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        node.run()
        rate.sleep()
