from fnmatch import fnmatch
import dlib
import matplotlib.pyplot as plt
import numpy as np
import os
from pyexpat import model
import fnmatch
import cv2
from cv_bridge import CvBridgeError


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
small_frame = cv2.imread("/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/src/lib/baggio.jpg")

# Detect the faces in the image
img_detected = detector(small_frame, 1)
img_shape = sp(small_frame, img_detected[0])
align_img = dlib.get_face_chip(small_frame, img_shape)
img_rep = np.array(model.compute_face_descriptor(align_img))

face_name = []
# Compare the face in real time with the faces that we already have
for i in range(0, len(img_rep)):
    name = "Face"
            
    for j in range(0, len(known_face)):
        euclidean_dist = list(np.linalg.norm(known_face - img_rep, axis=1) <= 0.6)
        if True in euclidean_dist:
            fst = euclidean_dist.index(True)
            name = known_name[fst]
            print("Face detected: " + name)
            break
    if name not in face_name:
        face_name.append(name)
print(face_name)
#print(face_name)
#bounding box
for i, rects in enumerate(img_detected):
    for name in face_name:    
        print(name)     
        cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (0, 255, 0), 2)
        cv2.putText(small_frame, name, (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the image
cv2.imshow('Video', small_frame)

cv2.waitKey(0)
    # Press q to quit
if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
        
    #close all open windows

    
