# Vision System

## Installation
**clone the repository**
```bash
git clone https://github.com/Nicolasalan/Perception-System.git
``` 

## **Usage**
**To start detection**
```bash
cd <catkin_workspace>/
source devel/setup.bash
```
**start detection**
```basg
roslaunch dodo_detector_ros detect_kinect.launch
```
**Follow this path in Rviz to see the detection in real time**
```
|-- rviz
    |-- add
        |-- by topic
            |-- dodo_detector_ros
                |-- labeled_image
                    |-- image
```
**Initialize object location**
```bash
roslaunch hera_objects objects.launch
```
**Call the service to see the positions**
```bash
rosservice call /objects "condition: 'closest'"
```