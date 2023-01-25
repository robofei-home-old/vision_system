#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf

from calendar import c
from pyexpat import model
import sys
import os
from turtle import back
from unicodedata import name

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32, Int32

from psutil import virtual_memory
from regex import E, F
import rospy
from sensor_msgs.msg import Image
import rospkg
import fnmatch
import cv2
import time
import face_recognition
from cv_bridge import CvBridge, CvBridgeError
from hera_face.srv import face_list
import dlib
import numpy as np


class Detector: 

   def __init__(self):
      image_topic = rospy.get_param('~image_topic')
      point_cloud_topic = rospy.get_param('~point_cloud_topic', None)

      self._global_frame = 'camera'
      self._frame = 'camera_depth_frame'
      self._tf_prefix = rospy.get_param('~tf_prefix', rospy.get_name())
      self._tf_listener = tf.TransformListener()
      # create detector
      self._bridge = CvBridge()

      # image and point cloud subscribers
      # and variables that will hold their values
      rospy.Subscriber(image_topic, Image, self.image_callback)

      if point_cloud_topic is not None:
         rospy.Subscriber(point_cloud_topic, PointCloud2, self.pc_callback)
      else:
         rospy.loginfo(
            'No point cloud information available. Objects will not be placed in the scene.')

      self._current_image = None
      self._current_pc = None

      # publisher for frames with detected objects
      self._imagepub = rospy.Publisher('~labeled_persons', Image, queue_size=10)

      self._tfpub = tf.TransformBroadcaster()
      rospy.loginfo('Ready to detect!')

   def image_callback(self, image):
      """Image callback"""
      # Store value on a private attribute
      self._current_image = image

   def pc_callback(self, pc):
      """Point cloud callback"""
      # Store value on a private attribute
      self._current_pc = pc

   def run(self):
           # run while ROS runs
      while not rospy.is_shutdown():
         # only run if there's an image present
         if self._current_image is not None:
            try:
               # if the user passes a fixed frame, we'll ask for transformation
               # vectors from the camera link to the fixed frame
               if self._global_frame is not None:
                       (trans, _) = self._tf_listener.lookupTransform('/' + self._global_frame,
                                                                  '/'+ self._frame, rospy.Time(0))

               small_frame = self._bridge.imgmsg_to_cv2(self._current_image, 'rgb8')
               time.sleep(1)  

               detector = dlib.get_frontal_face_detector()
               sp = dlib.shape_predictor("/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/src/shape_predictor_5_face_landmarks.dat")
               model  = dlib.face_recognition_model_v1("/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/src/dlib_face_recognition_resnet_model_v1.dat")
      
               self.people_dir = '/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/face_images/'
     
               files = fnmatch.filter(os.listdir(self.people_dir), '*.jpg')

               known_face = []
               known_name = []
               for f in range(0, len(files)):
                  for j in range(0,100):
                     img = dlib.load_rgb_image(self.people_dir + files[f])
                     img_detected = detector(img, 1)
                     img_shape = sp(img, img_detected[0])
                     align_img = dlib.get_face_chip(img, img_shape)
                     img_rep = np.array(model.compute_face_descriptor(align_img))
                     if len(img_detected) > 0:
                        known_face.append(img_rep)
                        known_name.append(files[f].split('.')[0])
                        break 
                     else:
                        rospy.loginfo("No face detected in image: " + files[f])
                        break

               #small_frame = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
               #time.sleep(1)   
               # Get a reference to webcam #0 (the default one)
               # We select bgr8 because its the OpneCV encoding by default
    
               # robot vision
               face_center = []
               face_name = []
               img_detected = detector(small_frame, 1)
               # se nao detectar ninguem
               if len(img_detected) == 0:
                  rospy.loginfo("No face detected")
                  #return '', 0.0, len(img_detected)
                  continue
               else:
                  print("Face Detectada", img_detected)
                  faces = dlib.full_object_detections()
            

                  for detection in img_detected:
                     faces.append(sp(small_frame, detection))
            
                  align_img = dlib.get_face_chips(small_frame, faces)                    
                  img_rep = np.array(model.compute_face_descriptor(align_img))

        #--------------------------------------------------------------------------------
    
                  for i in range(0, len(img_detected)):
                     name = 'Face'
                     for k in range(0, len(known_face)):      
                        euclidean_dist = list(np.linalg.norm(known_face - img_rep[i], axis=1) <= 0.6)
                        if True in euclidean_dist:
                           fst = euclidean_dist.index(True)
                           name = known_name[fst]
                        else:
                           continue
                    
                     face_name.insert(i, name)

            #--------------------------------------------------------------------------
                  coords = []
                  for i, rects in enumerate(img_detected):
                     lista = []
                     if face_name[i] in known_name:
                        cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (0, 255, 0), 2)
                        cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                     else:
                        cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (255, 0, 0), 2)
                        cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
      
                     center_x = (rects.right() + rects.left())/2
                     face_center.append(center_x)
                     lista.append(rects.left())
                     lista.append(rects.top())
                     lista.append(rects.right())
                     lista.append(rects.bottom())
                     coords.append(lista)

                     publish_tf = False
                     if self._current_pc is None:
                        rospy.loginfo(
                                   'No point cloud information available to track current object in scene')

                     # if there is point cloud data, we'll try to place a tf
                     # in the object's location
                     else:
                        y_center = round((rects.top() + rects.bottom())/2)
                        x_center = round((rects.left() + rects.right())/2)
                        # this function gives us a generator of points.
                        # we ask for a single point in the center of our object.
                        try:
                           pc_list = list(
                              pc2.read_points(self._current_pc,
                                                       skip_nans=True,
                                                       field_names=('x', 'y', 'z'),
                                                       uvs=[(x_center, y_center)]))
                        except:
                              continue
               

                        if len(pc_list) > 0:
                           publish_tf = True
                           # this is the location of our object in space
                           tf_id = str(face_name[i])

                           # if the user passes a tf prefix, we append it to the object tf name here
                           if self._tf_prefix is not None:
                              tf_id = self._tf_prefix + '/' + tf_id

                           tf_id = tf_id

                           point_x, point_y, point_z = pc_list[0]

                     # we'll publish a TF related to this object only once
                     if publish_tf:
                        # kinect here is mapped as camera_link
                        # object tf (x, y, z) must be
                        # passed as (z,-x,-y)
                        object_tf = [point_z, -point_x, -point_y]
                        frame = self._frame

                        # translate the tf in regard to the fixed frame
                        if self._global_frame is not None:
                              object_tf = np.array(trans) + object_tf
                              frame = self._global_frame

                        # this fixes #7 on GitHub, when applying the
                        # translation to the tf creates a vector that
                        # RViz just can'y handle
                        if object_tf is not None:
                           self._tfpub.sendTransform((object_tf),
                                             tf.transformations.quaternion_from_euler(
                                             0, 0, 0),
                                             rospy.Time.now(),
                                             tf_id,
                                             frame)
               cv2.destroyAllWindows()

            except CvBridgeError as e:
                print(e)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(e)


if __name__ == '__main__':
   rospy.init_node('people_launch', log_level=rospy.INFO)

   try:
      Detector().run()
   except KeyboardInterrupt:
      rospy.loginfo('Shutting down')
