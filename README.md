# Object Localization through Heuristic Search

Overview
--------

This stack provides two packages: kinect_sim and sbpl_perception.
The former is the depth-image renderer/simulator. Given a camera pose, set of 3D models/meshes and their locations,
it can generate a depth-image for that scene. 

The latter is the object localization package and contains the search environment that communicates with the planner.
It internally uses kinect_sim to generate 'states' for the planner to search through.

Setup
-----

1. Get ROS Hydro from http://wiki.ros.org/hydro/Installation/Ubuntu
2. Create a rosbuild workspace ~/hydro_workspace as described in http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment

```bash
git clone https://github.com/venkatrn/perception.git ~/hydro_workspace
git clone https://github.com/venkatrn/improved-mha-planner.git ~/hydro_workspace
rosmake sbpl_perception
#If above fails, try the following to diagnose error:
#rosmake sbpl
#rosmake kinect_sim
```
 If you have any missing ROS package dependencies, do:
 ```bash
 sudo apt-get install ros-hydro-<package> #or replace 'hydro' by appropriate version name
 ```
