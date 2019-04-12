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
from person_follower.msg import Location
from cv_bridge import CvBridge, CvBridgeError

sys.path.append("..")

class PersonFollow:
    def __init__(self):
        
        # Subscribers
        self.sub_img = rospy.Subscriber('/camera/color/image_rect_color', Image, self.imageCallback)
        self.sub_dep = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depthCallback)

        # Publishers
        self.pub_loc = rospy.Publisher('person_location', Location, queue=1)
        self.pub_debug = rospy.Publisher('detection_image', Image, queue=1)
        

        #member variables
        self.frameWidth = 640
        self.frameHeight = 480
        self.middlePixel = self.frameWidth/2.0
        self.raw_score = 0
        self.prev_score = 0
        self.pixelToNormalCoordsx = (2/self.frameWidth)

        self.bridge = CvBridge()
        self.msgread = {'image': False, 'depth': False}
        self.ready = False

        # Tunables
        self.alpha = 0.15 # score low pass filter
        self.turnGain = 5 # need tuned
        self.velocityGain = 5 # need tuned
        self.DEPTH_BOX_H = 0.10 * self.frameWidth # for averaging depth

        self.init()
        
    def imageCallback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.msgread['image'] = True
        except CvBridgeError as e:
            print(e)

    def depthCallback(self, msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.msgread['depth'] = True
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

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as self.sess:
                # Definite input and output Tensors for detection_graph
                self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = detection_graph.get_tensor_by_name(
                    'detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                self.detection_classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def get_depth(self, x_min, y_min, x_max, y_max):
        depth_roi = self.depth_img[y_max-self.DEPTH_BOX_H:y_max,
                                   x_min:x_max]
        dist = np.mean(depth_roi[~np.isinf(depth_roi)])
        return dist        

    def get_score(self, classes, scores):
        k = -1
        for i in range(classes.size):
            if classes[i] == 1 and scores[i] > .40:
                k = i
                return k, scores[i]

        return k, 0

    def handle_detection(self):
        image_np = self.img 

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        sz = 5
        boxes = np.squeeze(boxes)[:sz]
        classes = np.squeeze(classes).astype(np.int32)[:sz]
        scores = np.squeeze(scores)[:sz]

        k, self.raw_score = self.get_score(classes, scores)

        filt_score = (1-self.alpha)*self.prev_score + self.alpha*self.raw_score
        self.prev_score = filt_score

        loc = Location()
        if k == -1:
            loc.score = 0
            loc.dist = -1
            loc.left_edge = 0
            loc.right_edge = 0
        else:
            box = boxes[k]
            im_h, im_w, _ = image_np.shape
            corners = (int(box[1]*im_w), int(box[0]*im_h),
                       int(box[3]*im_w), int(box[2]*im_h))
            x_min = corners[0]
            y_min = corners[1]
            x_max = corners[2]
            y_max = corners[3]
            loc.score = filt_score
            loc.dist = self.get_depth(x_min, y_min, x_max, y_max)
            loc.left_edge = (x_min - self.frameWidth/2)*self.pixelToNormalCoordsx
            loc.right_edge = (x_max - self.frameWidth/2)*self.pixelToNormalCoordsx
            
        self.pub_loc.publish(loc)        
        
        if self.pub_debug.get_num_connections() > 0:
            img_debug = np.copy(image_np)
            if loc.score > 0.40:
                cv2.rectangle(img_debug, (corners[0], corners[1]), (corners[2], corners[3]), (255,0,0), 4)
            img_pub = self.bridge.cv2_to_imgmsg(img_debug)
            self.pub_debug.publish(img_pub)
        
    def run(self):
        if self.ready:
            self.handle_detection()
        else:
            self.check_ready()

if __name__ == '__main__':
    rospy.init_node('person_follow_node')
    node = PersonFollow()
    rate = rospy.Rate(15)
    try:
        while not rospy.is_shutdown():
            node.run()
            rate.sleep()
    except:
        rospy.ROSInterruptException
