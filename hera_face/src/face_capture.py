#!/usr/bin/env python3

import sys
import os
import rospy
from sensor_msgs.msg import Image
import rospkg
import cv2
import face_recognition
import numpy
import tf

from os.path import expanduser
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from hera_face.srv import face_capture

class FaceCapture():

    recog = 0

    def __init__(self):
        
        rospy.Service('face_captures', face_capture, self.handler)

        rospy.loginfo("Start FaceRecogniser Init process...")
        # get an instance of RosPack with the default search paths
        self.rate = rospy.Rate(5)
        rospack = rospkg.RosPack()
        # get the file path for my_face_recogniser
        self.path_to_package = rospack.get_path('hera_face')

        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.topic = "/usb_cam/image_raw"
        self._check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
        rospy.loginfo("Finished FaceRecogniser Init process...Ready")

        self._global_frame = "kinect_one_optical"
        # create detector
        self._bridge = CvBridge()

        rospy.loginfo('Ready to detect!')

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
        

    def recognise(self,data,request):

        # Get a reference to webcam #0 (the default one)
        try:
            # We select bgr8 because its the OpneCV encoding by default
            video_capture = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        
        # face_locations = []

        small_frame = cv2.resize(video_capture, (0, 0), fx=0.5, fy=0.5)

        image_path = os.path.join(self.path_to_package+'/face_images/')

        # Find all the faces and face encodings in the current frame of video
        print(face_recognition)
        face_locations = face_recognition.face_locations(small_frame)
       
        if not face_locations:
            rospy.logwarn("No Faces found, please get closer...")

        else:
            writeStatus = cv2.imwrite(str(image_path) + request.name + '.jpg', small_frame)
        
            if writeStatus is True:
                rospy.loginfo('Face '+request.name+' saved succeeded!')
                rospy.loginfo(str(image_path) + request.name + '.jpg')
                self.recog = 1
                return writeStatus
            else:
                rospy.loginfo('Face not saved!')
                return writeStatus
                

    def handler(self, request):
        self.recog = 0

        while self.recog == 0:
            resp = self.recognise(self.cam_image, request)
            self.rate.sleep()

            if resp is True:
                return "Salved!"
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('face_captures', log_level=rospy.INFO)
    FaceCapture()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
