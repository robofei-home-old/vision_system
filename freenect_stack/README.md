# **Freenect Stack**
Useful commands to use with Kinect without ROS:
freekinect_stack is a package that lets you use Kinect with ROS. To install the package, just follow these steps:
## Installation
Installing the freenect_stack package:
```bash
sudo apt-get install ros-$ROS_DISTRO-freenect-stack
```
## Usage
To use the package, simply run the following command:
```bash
roslaunch freenect_launch freenect.launch
```
Initially, the package will look for a Kinect connected to the computer. If it doesn't find any, it will show an error message. If found, it will launch Kinect and publish the topics:
```bash
/camera/rgb/image_raw
/camera/depth/image_raw
/camera/depth/points
```
## Usage with RViz
To view Kinect data in RViz, simply run the following command:
```bash
rviz
```
And add the topics:
```bash
/camera/rgb/image_raw
/camera/depth/image_raw
/camera/depth/points
```
All ready! Now just use the Kinect data in ROS.

## Use without ROS
To use Kinect without ROS, just follow these steps:
### Installation
Install the libfreenect package:
```bash
sudo apt-get install libfreenect-dev
```
### Usage
To use Kinect without ROS, simply run the following command:

**Depth with color**
```bash
freenect-cpp_pcview
```
![](/docs/image1.png)

**Depth plus RGB camera:**
```bash
freenect-cppview
```
![](/docs/image2.png)
**Expanded RGB camera:**
```bash
freenect-hiview
```
**Depth with RGB camera:**
```bash
freenect-regview
```
![](/docs/image3.png)
---
If you have a usb camera connected to your computer, follow the commands to use the usb camera:
## Installation
Install the usb_cam package:
```bash
sudo apt-get install ros-$ROS_DISTRO-usb-cam
```
## Usage
To use the package, simply run the following command:
```bash
roslaunch usb_cam usb_cam-test.launch
```
Initially, the package will look for a usb camera connected to the computer. If it doesn't find any, it will show an error message. If found, it will initialize the camera and publish the topic:
```bash
/camera/image_raw
```