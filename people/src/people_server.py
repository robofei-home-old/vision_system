#!/usr/bin/env python
import numpy as np
import rospy
import math
import sys
import cv2
import tf

import tf2_ros.transform_broadcaster

from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image
from people.msg import People, Person, Keypoint
from people.srv import Pause, PauseResponse
from people.srv import Once, OnceResponse
from geometry_msgs.msg import TransformStamped, Pose, Point

sys.path.append('/usr/local/python/');
from openpose import pyopenpose as op

class PeopleServer():
    """docstring for people."""
    def __init__(self):

        # init
        self.rate = rospy.Rate(10)
        self.bridge = CvBridge()
        self.br = tf2_ros.transform_broadcaster.TransformBroadcaster()

        # get params
        self.models = rospy.get_param('~models',None)
        self.image_color = rospy.get_param('~image_color',None)
        self.image_cloud = rospy.get_param('~image_cloud',None)
        self.pause = rospy.get_param('~pause',True)

        # self.x = []
        # self.y = []

        # set keypoint names
        self.keypoint_names = ['Nose','Neck',
            'RShoulder','RElbow','RWrist',
            'LShoulder','LElbow','LWrist',
            'MidHip',
            'RHip','RKnee','RAnkle',
            'LHip','LKnee','LAnkle',
            'REye','LEye','REar','LEar',
            'LBigToe','LSmallToe','LHeel',
            'RBigToe','RSmallToe','RHeel',
            'Background']

        # data
        self.color = None
        self.cloud = None

        # Subscribers
        rospy.Subscriber(self.image_color, Image, self.callback_color)
        if(self.image_cloud): rospy.Subscriber(self.image_cloud, PointCloud2, self.callback_cloud)

        # Publishers
        self.openpose_pub = rospy.Publisher("people_image", Image, queue_size=10)
        self.people_pub = rospy.Publisher("people", People, queue_size=10)

        # Services
        rospy.Service('~pause', Pause, self.handle_pause)
        rospy.Service('~once', Once, self.handle_once)

        # Starting OpenPose
        params = dict()
        params['model_folder'] = self.models
        # params["face"] = True
        # params["hand"] = True
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        rospy.loginfo("People sensor is {}!".format("paused" if self.pause else "running") )

        # start main loop
        self.loop()

    def take_distance(self, elem):
        p = np.array([elem.pose.position.x,
                      elem.pose.position.y,
                      elem.pose.position.z])
        return np.linalg.norm(p)

    def people_msg(self, cloud, datum, stamp):
        msg = People()
        msg.header.stamp = stamp

        # get person
        if(datum.poseKeypoints.shape and
           datum.poseKeypoints.shape[0] > 0):
           for id, keypoints in enumerate(datum.poseKeypoints):
               person = Person()
               person.name = 'Person_'+str(id)

               bodypoints = []
               #####################
               for key, point in enumerate(keypoints):
                   try:
                       point_list = list(pc2.read_points(cloud, skip_nans=True,
                                         field_names=('x', 'y', 'z'),
                                         uvs=[(int(point[0]), int(point[1]))]))
                   except Exception as e:
                       print "error on read points"
                       continue

                   bodypoint = Keypoint()
                   if(len(point_list) > 0 ):
                        bodypoint.name = self.keypoint_names[key]
                        bodypoint.position2d.x = int(point[0])
                        bodypoint.position2d.y = int(point[1])
                        bodypoint.position3d.position.x = point_list[0][2]
                        bodypoint.position3d.position.y = -point_list[0][0]
                        bodypoint.position3d.position.z = -point_list[0][1]
                   bodypoints.append(bodypoint)
               #####################

               # 1 = neck
               person.pose.position.x = bodypoints[1].position3d.position.x
               person.pose.position.y = bodypoints[1].position3d.position.y
               person.pose.position.z = bodypoints[1].position3d.position.z
               person.pose.orientation.w = 1.0

               # append person
               msg.people.append(person)

        # sort by distance
        msg.people.sort(key=self.take_distance)

        return msg

    def print_tfs(self, people, stamp):
        for person in people:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "camera_link"
            t.child_frame_id = person.name
            t.transform.translation = person.pose.position
            t.transform.rotation = person.pose.orientation
            self.br.sendTransform([t])

    def run(self):
        stamp = rospy.Time.now()
        color = self.color
        cloud = self.cloud

        if( color is None):
            raise Exception('there is no color data')
        if( cloud is None):
            raise Exception('there is no point cloud data')

        # OpenPose Process
        datum = op.Datum()
        datum.cvInputData = color
        self.opWrapper.emplaceAndPop([datum])



        # Publish OpenPose Image
        img = datum.cvOutputData
        op_frame = self.bridge.cv2_to_imgmsg(img, "bgr8")
        op_frame.header.stamp = stamp
        self.openpose_pub.publish(op_frame)

        # get people msg
        msg = self.people_msg(cloud, datum, stamp)

        # publish people msg
        self.people_pub.publish(msg)

        # publish tfs
        self.print_tfs(msg.people, stamp)

        return msg.people


    def callback_color(self, data):
        self.color = self.bridge.imgmsg_to_cv2(data, 'bgr8')

    def callback_cloud(self, data):
        self.cloud = data

    def handle_pause(self, req):
        if(self.pause == False and req.pause == False):
            self.pause = req.pause
            rospy.loginfo("People sensor is already running!" )
            return RunResponse(False)
        elif(self.pause == False and req.pause == True):
            self.pause = req.pause
            rospy.loginfo("People sensor is now paused!" )
            return RunResponse(True)
        elif(self.pause == True and req.pause == False):
            self.pause = req.pause
            rospy.loginfo("People sensor is now running!" )
            return RunResponse(True)
        elif(self.pause == True and req.pause == True):
            self.pause = req.pause
            rospy.loginfo("People sensor is already paused!")
            return RunResponse(False)

    def handle_once(self, req):
        try:
            people = self.run()
            return OnceResponse(people)
        except Exception as e:
            print e
            return OnceResponse([])

    def loop(self):
        while not rospy.is_shutdown():
            if not self.pause:
                self.rate.sleep()
                try:
                    people = self.run()
                except Exception as e:
                    print e
