#!/usr/bin/env python3

from pyexpat import model
import sys
import os

from cv2 import VideoCapture
import rospy
from sensor_msgs.msg import Image
import rospkg
import fnmatch
import cv2
from cv_bridge import CvBridge, CvBridgeError
from hera_face.srv import face_list
import dlib
import numpy as np

class FacesFrames(object):
    # cuidado para nao ter imagem com tamanhos diferentes ou cameras diferentes, pois o reconhecimento nao vai funcionar

     def __init__(self):
        
          self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.camera_callback)

          rospack = rospkg.RosPack()
          # get the file path for my_face_recogniser
          self.path_to_package = rospack.get_path('hera_face')

          self.bridge_object = CvBridge()
          self.detector = dlib.get_frontal_face_detector()
          self.sp = dlib.shape_predictor("/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/src/shape_predictor_5_face_landmarks.dat")
          self.model  = dlib.face_recognition_model_v1("/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/src/dlib_face_recognition_resnet_model_v1.dat")

          self.people_dir = '/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/face_images/'

          self.files = fnmatch.filter(os.listdir(self.people_dir), '*.jpg')

          self.known_face = []
          self.known_name = []
          self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
          rospy.loginfo("Finished Faces Frames Init process...Ready")

          video_capture = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
                


          for f in range(0, len(self.files)):
               for j in range(0,100):
                    img = dlib.load_rgb_image(self.people_dir + self.files[f])
                    img_detected = self.detector(img, 1)
                    img_shape = self.sp(img, img_detected[0])
                    align_img = dlib.get_face_chip(img, img_shape)
                    img_rep = np.array(self.model.compute_face_descriptor(align_img))
                    if len(img_detected) > 0:
                         self.known_face.append(img_rep)
                         self.known_name.append(self.files[f].split('.')[0])
                         break 
                    else:
                         rospy.loginfo("No face detected in image: " + self.files[f])
                         break

          small_frame = cv_image
          #small_frame = VideoCapture(1)
          while small_frame.isOpened():
               face_center = []
               face_name = []
               img_detected = self.detector(small_frame, 1)
               faces = dlib.full_object_detections()
               for detection in img_detected:
                    faces.append(self.sp(small_frame, detection))
                    
               align_img = dlib.get_face_chips(small_frame, faces)                    
               img_rep = np.array(self.model.compute_face_descriptor(align_img))
          
               for i in range(0, len(img_detected)):
                    name = 'Face'
                    for k in range(0, len(self.known_face)):      
                         euclidean_dist = list(np.linalg.norm(self.known_face - img_rep[i], axis=1) <= 0.6)
                         if True in euclidean_dist:
                              fst = euclidean_dist.index(True)
                              name = self.known_name[fst]
                         
                    face_name.insert(i, name)

               for i, rects in enumerate(img_detected):

                    if face_name[i] in self.known_name:
                         cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (0, 255, 0), 2)
                         cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                         cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (255, 0, 0), 2)
                         cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    center_x = (rects.right() + rects.left())/2
                    face_center.append(center_x)
                    
               window = dlib.image_window()
               window.set_image(small_frame)
               cv2.imwrite('/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/face_recogs/recog.jpg', small_frame)
               xmax = rects.right()
               xmin = rects.left()
               ymax = rects.bottom()
               ymin = rects.top()
               rospy.loginfo("xmax: " + str(xmax))
               rospy.loginfo("xmin: " + str(xmin))
               rospy.loginfo("ymax: " + str(ymax))
               rospy.loginfo("ymin: " + str(ymin))
               print(self.face_name)
               # transformar o center_x em float 
               cv2.imshow("Faces", small_frame)
               cv2.waitKey(1)

     def _check_cam_ready(self):
          self.cam_image = None
          while self.cam_image is None and not rospy.is_shutdown():
               try:
                    self.cam_image = rospy.wait_for_message(self.topic, Image, timeout=1.0)
                    rospy.logdebug("Current "+self.topic+" READY=>" + str(self.cam_image))

               except:
                    rospy.logerr("Current "+self.topic+" not ready yet, retrying.")

     def camera_callback(self,data):
          self.cam_image = data

     def recog(self,data):
          
          try:
               # We select bgr8 because its the OpenCV encoding by default
               cv_image = self.bridge_object.imgmsg_to_cv2(0, desired_encoding="bgr8")
          except CvBridgeError as e:
               print(e)

               for f in range(0, len(self.files)):
                    for j in range(0,100):
                         img = dlib.load_rgb_image(self.people_dir + self.files[f])
                         img_detected = self.detector(img, 1)
                         img_shape = self.sp(img, img_detected[0])
                         align_img = dlib.get_face_chip(img, img_shape)
                         img_rep = np.array(self.model.compute_face_descriptor(align_img))
                         if len(img_detected) > 0:
                              self.known_face.append(img_rep)
                              self.known_name.append(self.files[f].split('.')[0])
                              break 
                         else:
                              rospy.loginfo("No face detected in image: " + self.files[f])
                              break

               #small_frame = cv_image
               small_frame = VideoCapture(0)
               while small_frame.isOpened():
                    face_center = []
                    face_name = []
                    img_detected = self.detector(small_frame, 1)
                    faces = dlib.full_object_detections()
                    for detection in img_detected:
                         faces.append(self.sp(small_frame, detection))
                         
                    align_img = dlib.get_face_chips(small_frame, faces)                    
                    img_rep = np.array(self.model.compute_face_descriptor(align_img))
               
                    for i in range(0, len(img_detected)):
                         name = 'Face'
                         for k in range(0, len(self.known_face)):      
                              euclidean_dist = list(np.linalg.norm(self.known_face - img_rep[i], axis=1) <= 0.6)
                              if True in euclidean_dist:
                                   fst = euclidean_dist.index(True)
                                   name = self.known_name[fst]
                              
                         face_name.insert(i, name)

                    for i, rects in enumerate(img_detected):

                         if face_name[i] in self.known_name:
                              cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (0, 255, 0), 2)
                              cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                         else:
                              cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (255, 0, 0), 2)
                              cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                         center_x = (rects.right() + rects.left())/2
                         face_center.append(center_x)
                         
                    window = dlib.image_window()
                    window.set_image(small_frame)
                    cv2.imwrite('/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/face_recogs/recog.jpg', small_frame)
                    xmax = rects.right()
                    xmin = rects.left()
                    ymax = rects.bottom()
                    ymin = rects.top()
                    rospy.loginfo("xmax: " + str(xmax))
                    rospy.loginfo("xmin: " + str(xmin))
                    rospy.loginfo("ymax: " + str(ymax))
                    rospy.loginfo("ymin: " + str(ymin))
                    face_names_str = " - ".join(face_name)
                    # transformar o center_x em float 
                    face_center_float = [float(i) for i in face_center]
                    cv2.imshow("Faces", small_frame)
                    cv2.waitKey(1)

            
if __name__ == '__main__':
     rospy.init_node('face_frames', log_level=rospy.INFO)
     FacesFrames()

     try:
         rospy.spin()
     except rospy.ROSInterruptException:
         pass

