#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
import rospy
import imutils
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from hera_face.srv import color_detect


class ColorFilter():
    recog = 0

    def __init__(self):
        rospy.Service('color_filter', color_detect, self.handler)
        
        rospy.loginfo("Start Color Filter Init process...")
        # get an instance of RosPack with the default search paths
        self.rate = rospy.Rate(5)

        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.topic = "/usb_cam/image_raw"
        self._check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
        rospy.loginfo("Finished Color Filter Init process...Ready")

        os.system("v4l2-ctl -d /dev/video0 -c focus_auto=0 && v4l2-ctl -d /dev/video0 -c focus_absolute=0")
        os.system("v4l2-ctl -d /dev/video0 -c saturation=50")
        os.system("v4l2-ctl -d /dev/video0 -c brightness=100")

    def _check_cam_ready(self):
      self.cam_image = None
      while self.cam_image is None and not rospy.is_shutdown():
         try:
               self.cam_image = rospy.wait_for_message(self.topic, Image, timeout=1.0)
               rospy.logdebug("Current " + self.topic + " READY=>" + str(self.cam_image))

         except:
               rospy.logerr("Current " + self.topic+" not ready yet, retrying.")

    def camera_callback(self,data):
        self.cam_image = data

    def filter(self, data):

        try:
            # We select bgr8 because its the OpneCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        #I resized the image so it can be easier to work with
        image = cv2.resize(cv_image,(640,480))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # red color boundaries [B, G, R]
        lower = [0, 0, 126]
        upper = [179, 26, 255]

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        ret, thresh = cv2.threshold(mask, 40, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            # draw in blue the contours that were founded
            cv2.drawContours(output, contours, -1, 255, 3)

            # find the biggest countour (c) by the area
            c = max(contours, key = cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # draw the biggest contour (c) in green
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
            # printar o centro do contorno
            center_x = x + w/2
            center_y = y + h/2
        
            return float(center_x), float(center_y)
        else:
            return 0.0, 0.0

    def handler(self, request):
        self.recog = 0

        if request.name == '':
            while self.recog == 0:
                cx, cy = self.filter(self.cam_image)
                self.rate.sleep()
                #rospy.loginfo("center x:", cx)
                #rospy.loginfo("center y:", cy)

                return float(cx), float(cy)
        else:
            return 0.0, 0.0


if __name__ == '__main__':
    rospy.init_node('color_filter', log_level=rospy.INFO)
    ColorFilter()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
