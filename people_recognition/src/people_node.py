#!/usr/bin/env python

import rospy
from people_server import PeopleServer

if __name__ == "__main__":
    rospy.init_node('people_node')
    try:
        PeopleServer()
    except KeyboardInterrupt:
        print("Shutting down")
