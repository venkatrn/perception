# PERCH: Perception via Search for Multi-Object Recognition and Localization

Overview
--------
This stack provides the implementation for PERCH (described in TODO: add link).

The package 'kinect_sim' is based-off PCL's simulation API and serves as the depth-image renderer/simulator for PERCH. This provides the functionality to generate the depth-image for a scene, given a camera pose, set of 3D models/meshes and their poses.

The 'sbpl_perception' package implements the PERCH algorithm. This works in conjunction with the SBPL planning library (https://github.com/sbpl/sbpl) and provides the 'search environment'. The documentation in the SBPL github page provides examples of the planner-environment usage.  

Setup
-----

1. Get ROS Hydro (or Indigo) from http://wiki.ros.org/hydro/Installation/Ubuntu
2. Create a rosbuild workspace ~/hydro_workspace as described in http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment and make sure ~/hydro_workspace/sandbox is in your ROS package path.

```bash
cd ~/hydro_workspace/sandbox && git clone https://github.com/venkatrn/perception.git
cd ~/hydro_workspace/sandbox && git clone https://github.com/venkatrn/improved-mha-planner.git 
rosmake sbpl_perception
#If above fails, try the following to diagnose error:
#rosmake sbpl
#rosmake kinect_sim
```
 If you have any missing ROS package dependencies, do:
 ```bash
 sudo apt-get install ros-hydro-<package> #or replace 'hydro' by appropriate version name
 ```

Test
----

Download required data files from https://sbpl.pc.cs.cmu.edu/shared/Venkat/sbpl_perception/ and place the data directory under ~/hydro_workspace/sandbox/perception/sbpl_perception/

Example usage:
```bash
cd sbpl_perception && mkdir visualization
roslaunch sbpl_perception experiments.launch 
```

The states 'expanded' as well as the goal state will be available in sbpl_perception/visualization. To also visualize the 'generated' states, 

```bash
roslaunch sbpl_perception experiments.launch image_debug:=true
```

