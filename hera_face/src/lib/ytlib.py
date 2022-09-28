from re import A
import dlib
import matplotlib.pyplot as plt
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("/home/sousa/Documents/catkin_hera/src/vision_system/hera_face/src/lib/shape_predictor_5_face_landmarks.dat")
model  = dlib.face_recognition_model_v1("/home/sousa/Documents/catkin_hera/src/vision_system/hera_face/src/lib/dlib_face_recognition_resnet_model_v1.dat")

img1 = dlib.load_rgb_image("/home/sousa/Documents/catkin_hera/src/vision_system/hera_face/src/lib/face_images/Hunter.jpg")
img2 = dlib.load_rgb_image("/home/sousa/Documents/catkin_hera/src/vision_system/hera_face/src/lib/face_images/quadrado.jpg")

img1_detected = detector(img1, 1)
img2_detected = detector(img2, 1)

img1_shape = sp(img1, img1_detected[0])
img2_shape = sp(img2, img2_detected[0])

align_img1 = dlib.get_face_chip(img1, img1_shape)
align_img2 = dlib.get_face_chip(img2, img2_shape)

img1_rep = np.array(model.compute_face_descriptor(align_img1))
img2_rep = np.array(model.compute_face_descriptor(align_img2))

def euclidian(src_rep, test_rep):
    x = src_rep - test_rep
    x = np.sum(np.multiply(x,x))
    x = np.sqrt(x)
    return x

dist = euclidian(img1_rep, img2_rep)

th = 0.6

if dist < th:
    print("Are the same")
else: 
    print("Are diff")