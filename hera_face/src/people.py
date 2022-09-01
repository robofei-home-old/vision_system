#!/usr/bin/env python3
import sys
import os
import rospy
from sensor_msgs.msg import Image
import rospkg
import fnmatch
import cv2
import time
import face_recognition
from cv_bridge import CvBridge, CvBridgeError
from hera_face.srv import face_list

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

    def recognise(self,data):

        # Get a reference to webcam #0 (the default one)
        try:
            # We select bgr8 because its the OpneCV encoding by default
            video_capture = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        
        self.people_dir = '/home/hera/catkin_hera/src/3rdParty/vision_system/hera_face/face_images/'

        files = fnmatch.filter(os.listdir(self.people_dir), '*.jpg')

        faces_images = []
        known_face_encodings = []
        known_face_names = []
        for i in range(0,len(files)):

            for j in range(0,100):            
                name = files[i].replace('.jpg','')
                face = face_recognition.load_image_file(self.people_dir + files[i])
                enc = face_recognition.face_encodings(face)
                if(len(enc)):
                    known_face_names.append(name)
                    faces_images.append(face)        
                    known_face_encodings.append(enc[0])
                    break;
                else:
                    rospy.logerr("não achei cara nenhuma depois de 100 tentativas!!")


        # robot vision
        small_frame = cv2.resize(video_capture, (0, 0), fx=0.5, fy=0.5)

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        face_center = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Face"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Draw a box around the face
            cv2.rectangle(video_capture, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(video_capture, (left, bottom - 35), (right, bottom), (0, 0, 255))
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(video_capture, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            height, width = video_capture.shape[:2]

            center_x = (right + left)/2
            #print("posicao em x:",center_x)
            face_center.append(center_x)

        face_names_str = " - ".join(face_names)
        # transformar o center_x em float 
        face_center_float = [float(i) for i in face_center]
        #rospy.loginfo("people: " + face_names_str)
        
        # Display the resulting image
        #cv2.imshow("Image window", video_capture)
        #cv2.waitKey(1)
        # retornar o x da pessoa especifical na imagem
        # input name of the person Outout x of the person and person name


        self.recog = 1
        print(face_center)
        return face_names, face_center_float


    def handler(self, request):
        self.recog = 0
        recog_request = 0

        if request.name == '':

            while self.recog == 0:
                resp, center = self.recognise(self.cam_image)
                self.rate.sleep()

                return resp, (str(center))

        else:
            # retornar somente o nome da pessoa e a posciao em center_x
            while recog_request == 0:
                resp, center = self.recognise(self.cam_image)
                self.rate.sleep()

                for i in resp:
                    j = resp.index(i)
                    print(i)
                    if i == request.name:
                        recog_request  = 1
                        print("request ",resp)
                        print("center ",center[j])

                        return i, center[j]

        cv2.destroyAllWindows()
        


if __name__ == '__main__':
    rospy.init_node('face_recog', log_level=rospy.INFO)
    FaceRecog()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
