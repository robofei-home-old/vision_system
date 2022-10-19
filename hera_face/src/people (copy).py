#!/usr/bin/env python3
from pyexpat import model
import sys
import os
from turtle import back
from unicodedata import name
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

class FaceRecog():
    # cuidado para nao ter imagem com tamanhos diferentes ou cameras diferentes, pois o reconhecimento nao vai funcionar

    recog = 0

    def __init__(self):
        rospy.Service('face_recog', face_list, self.handler)
        
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

    def recognise(self, data, nome_main):

        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor("/home/hera/catkin_hera/src/3rdParty/vision_system/hera_face/src/shape_predictor_5_face_landmarks.dat")
        model  = dlib.face_recognition_model_v1("/home/hera/catkin_hera/src/3rdParty/vision_system/hera_face/src/dlib_face_recognition_resnet_model_v1.dat")

        self.people_dir = '/home/hera/catkin_hera/src/3rdParty/vision_system/hera_face/face_images/'

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

        small_frame = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        time.sleep(1)   
    # Get a reference to webcam #0 (the default one)
        # We select bgr8 because its the OpneCV encoding by default
    
        # robot vision
        face_center = []
        face_name = []
        img_detected = detector(small_frame, 1)
        if len(img_detected) > 0:
            
            faces = dlib.full_object_detections()
            for detection in img_detected:
                faces.append(sp(small_frame, detection))
            
            align_img = dlib.get_face_chips(small_frame, faces)                    
            img_rep = np.array(model.compute_face_descriptor(align_img))
    
            for i in range(0, len(img_detected)):
                name = 'Face'
                for k in range(0, len(known_face)):      
                    euclidean_dist = list(np.linalg.norm(known_face - img_rep[i], axis=1) <= 0.6)
                    if True in euclidean_dist:
                        fst = euclidean_dist.index(True)
                        name = known_name[fst]
                    
                face_name.insert(i, name)

            for i, rects in enumerate(img_detected):

                if face_name[i] in known_name:
                    cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (0, 255, 0), 2)
                    cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (255, 0, 0), 2)
                    cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                center_x = (rects.right() + rects.left())/2
                face_center.append(center_x)
            
            window = dlib.image_window()
            window.set_image(small_frame)
            cv2.imwrite('/home/hera/catkin_hera/src/3rdParty/vision_system/hera_face/face_recogs/recog.jpg', small_frame)
            #k = cv2.waitKey(0)
            #if k == 27:         # wait for ESC key to exit    cv2.destroyAllWindows()
            #    cv2.destroyAllWindows()
            face_names_str = " - ".join(face_name)
            # transformar o center_x em float 

            # self.recog = 1
            print("Face Recognised: ", face_name)
            print("Face centers: ", face_center)
            print("Pessoas na foto: ", len(img_detected))
            if nome_main == '':
                name = 'face'
                center = '0.0'
                self.recog = 1
                return name, center, len(img_detected)
            elif nome_main in face_name:
                center = face_center[face_name.index(nome_main)]
                name = face_name[face_name.index(nome_main)]
                #face_center_float = [float(i) for i in face_center]
                print("Pessoa encontrada")
                self.recog = 1
                return name, center, len(img_detected)
        else:
            time.sleep(1)
            print("No face detected")
            


    def handler(self, request):
        self.recog = 0
        recog_request = 0

        if request.name == '':

            while self.recog == 0:
                self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
                
                name, center, num = self.recognise(self.cam_image, request.name)
                self.rate.sleep()
                

                return name, float(center), num

        else:
            # retornar somente o nome da pessoa e a posciao em center_x
            while self.recog == 0:
                
                self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
                
                name, center, num = self.recognise(self.cam_image, request.name)
                self.rate.sleep()
                

                return name, float(center), num

        cv2.destroyAllWindows()
        


if __name__ == '__main__':
    rospy.init_node('face_recog', log_level=rospy.INFO)
    FaceRecog()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
