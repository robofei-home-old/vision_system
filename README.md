# Vision System

## Installation
**Clone the repository**
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
Before starting the vision system, it is necessary to modify the main_config.yaml file so that the system can find the configuration files. To do this, just modify the path to the vision system configuration file. The vision system configuration file is located at:
```bash
<catkin_workspace>/src/Perception-System/dodo_detector_ros/config/main_config.yaml
```
The main parameters that must be modified are:
```yaml
# ['sift', 'rootsift', 'tf1', 'tf2']
detector_type: tf2

saved_model: ~/dodo_detector_ros/saved_model

# path of file saved_model.pb
inference_graph: ~/dodo_detector_ros/saved_model/saved_model.pb

# file path label_map.pbtxt
label_map: ~/dodo_detector_ros/saved_model/label_map.pbtxt

# minimum confidence for detection
tf_confidence: 0.9 

sift_min_pts: 10

sift_database_path: ~/.dodo_detector_ros/sift_database

# local frame where detection will be performed
global_frame: kinect_one_depth
```
After modifying the main_config.yaml file, just start the system view with the command:
```bash
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
To initialize the object location system, you first have to modify the objects.py file so that the coordinates have a local reference. To do this, just modify the objects.py file, which is located at:
```bash
<catkin_workspace>/src/Perception-System/hera_objects/src/objects.py
```
Inside the file there is a parameter called `reference_frame`, which only places the frame in which the object will have as a reference. After modifying the file, just start the object location system with the command:
```bash
roslaunch hera_objects objects.launch
```
**Call the service to see the positions**
```bash
rosservice call /objects "condition: 'closest'"
```
Example of output:
```bash
position: 
  x: 0.0
  y: 0.0
  z: 0.0
  rx: 0.0
  ry: 0.0
  rz: 0.0
```
