# Deliberative Perception for Multi-Object Recognition and Localization

Overview
--------
This library provides implementations for single and multi-object instance localization from RGB-D sensor (MS Kinect, ASUS Xtion etc.) data. These are based on the <a href="http://www.cs.cmu.edu/~venkatrn/papers/icra16a.pdf">PERCH (Perception via Search)</a> and <a href="http://www.cs.cmu.edu/~venkatrn/papers/rss16.pdf">D2P (Discriminatively-guided Deliberative Perception)</a> algorithms.

Requirements
------------
- Ubuntu 16.04+
- ROS Kinetic (active development only on Kinetic)

Setup (For running with a robot camera or bagfile recorded from robot)
-----
1. Create a catkin_ws and clone the following (clone realsense package to work with real camera) :
```
https://github.com/SBPL-Cruz/improved-mha-planner
https://github.com/venkatrn/sbpl_utils.git
https://github.com/IntelRealSense/realsense-ros
```
2. Install Open CV 2.4 if not already installed. You can follow steps on the <a href="https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html">Open CV website</a>
3. Install gsl, vtk library :
```
sudo apt-get install libgsl-dev libvtk6-dev
```
4. Compile the packages in the catkin_ws
5. If you get compilation errors in ```octree_pointcloud_changedetector.h```, follow steps <a href="https://github.com/PointCloudLibrary/pcl/issues/2564">here</a> to fix
6. Check parameters (frame names etc.) in the launch file :
```
object_recognition_node/launch/roman_object_recognition_robot.launch
```
7. Check camera parameters in :
```
sbpl_perception/config/roman_camera_config.yaml
```
7. Launch camera and code using (the transforms should be being published by another code or bag file, dont launch camera if using bag file) : 
```
roslaunch object_recognition_node roman_object_recognition_robot.launch urdf:=false
roslaunch realsense2_camera rs_rgbd.launch camera:=/head_camera publish_tf:=false
```
8. To test you can download sample bag file from this <a href="https://drive.google.com/file/d/1X4yzLiQTnaXYLKMgNcFwvKDNLZDHyxPz/view?usp=sharing">link</a>
