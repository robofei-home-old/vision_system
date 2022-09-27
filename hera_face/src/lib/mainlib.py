from fnmatch import fnmatch
from re import A
import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from pyexpat import model
import fnmatch
import cv2

def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/src/lib/shape_predictor_5_face_landmarks.dat")
model  = dlib.face_recognition_model_v1("/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/src/lib/dlib_face_recognition_resnet_model_v1.dat")

people_dir = "/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/src/lib/face_images/"
files = fnmatch.filter(os.listdir(people_dir), '*.jpg')

# Load all the images that we want to compare and already have
images = []
known_face = []
known_name = []

for f in range(0, len(files)):
    for j in range(0,100):
        img = dlib.load_rgb_image(people_dir + files[f])
        img_detected = detector(img, 1)
        img_shape = sp(img, img_detected[0])
        align_img = dlib.get_face_chip(img, img_shape)
        img_rep = np.array(model.compute_face_descriptor(align_img))
        if len(img_detected) > 0:
            known_face.append(img_rep)
            known_name.append(files[f].split('.')[0])
            break
        else:
            print("No face detected in image: " + files[f])
            break
# Load the image in real time, have to change the path to de video capture
small_frame = cv2.imread("/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/src/lib/face_images/teste.jpg")

# Detect the faces in the image
img_detected = detector(small_frame, 1)
img_shape = sp(small_frame, img_detected[0])
align_img = dlib.get_face_chip(small_frame, img_shape)
img_rep = np.array(model.compute_face_descriptor(align_img))

# Compare the face in real time with the faces that we already have
aux = 1
for i in range(0, len(known_face)):
    #euclidian distance
    euclidean_dist = euclidean(known_face[i], img_rep)

    if euclidean_dist < 0.6 and euclidean_dist < aux:
        aux = euclidean_dist
        face = known_face[i]
        name = known_name[i]
        print("Face detected: " + known_name[i])

#bounding box
for face in img_detected:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(small_frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the image
    cv2.imshow('Video', small_frame)

    cv2.waitKey(0)
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        
    #close all open windows

    
