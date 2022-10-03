#from fnmatch import fnmatch
import dlib
import matplotlib.pyplot as plt
import numpy as np
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
files = fnmatch.filter(os.listdir(people_dir), '*')

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

print(known_name)
# Load the image in real time, have to change the path to de video capture
#small_frame = cv2.imread("/home/robofei/catkin_hera/src/3rdParty/vision_system/hera_face/src/lib/face_images/teste.jpg")

#open camera   
video_capture = cv2.VideoCapture(0)
while video_capture.isOpened():
    _,small_frame = video_capture.read()
    # Detect the faces in the image
    try:
        face_name = []
        img_detected = detector(small_frame, 1)
        print(img_detected)
        #img_shape = sp(small_frame, img_detected[0])
        faces = dlib.full_object_detections()
        for detection in img_detected:
            faces.append(sp(small_frame, detection))
        align_img = dlib.get_face_chips(small_frame, faces)
        #print("MATRIZ:", align_img)
        img_rep = np.array(model.compute_face_descriptor(align_img))
        #print("face RECOG: ", len(img_rep))

        # Compare the face in real time with the faces that we already have 
        #aux = 1 
        #compare each face in the image with the faces in the database
        for i in range(0, len(img_detected)):
            name = 'Face'
            #print(align_img)
            #for align_imgs in align_img:
            #align_img = cv2.resize(align_img, (0, 0), fx=3.33, fy=3.33)
            for k in range(0, len(known_face)):      
                euclidean_dist = list(np.linalg.norm(known_face - img_rep[i], axis=1) <= 0.6)
                print(euclidean_dist)
                if True in euclidean_dist:
                    fst = euclidean_dist.index(True)
                    print(fst)
                    name = known_name[fst]
                
            face_name.insert(i, name)
        print(face_name)
        #print(face_name)
        #bounding box
        for i, rects in enumerate(img_detected):
            center_x = int((rects.left() + rects.right()) / 2)
            
            cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (0, 255, 0), 2)
            print(rects.left(), rects.top(), rects.right(), rects.bottom())
            cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
    except:
        pass
    # Show the image
    cv2.imshow('Video', small_frame)

    cv2.waitKey(5) & 0xFF == ord('q')


video_capture.release()
cv2.destroyAllWindows()
        #close all open windows

    
