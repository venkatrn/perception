# Deliberative Perception for Multi-Object Recognition and Localization

Overview
--------
This repository provides the implementations for <a href="http://www.cs.cmu.edu/~venkatrn/papers/icra16a.pdf">PERCH (Perception via Search)</a> and <a href="http://www.cs.cmu.edu/~venkatrn/papers/rss16.pdf">D2P (Discriminatively-guided Deliberative Perception)</a>, algorithms for multi-object recognition and localization in RGB-D images. In addition, it contains a number of useful tools for dealing with RGB-D images.

The package 'kinect_sim' is based on PCL's simulation API and serves as the depth-image renderer/simulator for PERCH. This provides the functionality to generate the depth-image for a scene, given a camera pose, set of 3D models/meshes and their poses.

The 'sbpl_perception' package implements the PERCH and D2P algorithms. Internally, this works in conjunction with the SBPL planning library (https://github.com/sbpl/sbpl) which provides several domain-independent graph-search algorithms.

Dependencies
------------

- ROS Hydro+
- C++11
- OpenCV 2.x
- PCL 1.7+

Setup
-----

1. Get ROS Hydro (or Indigo) from http://wiki.ros.org/hydro/Installation/Ubuntu
2. Set up a catkin workspace ~/my_workspace (http://wiki.ros.org/catkin/Tutorials/create_a_workspace).
3. Download the <a href="https://raw.githubusercontent.com/venkatrn/perception/master/perch.rosinstall" download="perch.rosinstall">rosinstall file</a> to your workspace.

```bash
cd ~/my_workspace
wstool init src
wstool merge -t src perch.rosinstall
wstool update -t src
catkin_make -DCMAKE_BUILD_TYPE=Release
```

Demo
----
First, download the RGB-D dataset and object CAD models (92 MB total):

```bash
roscd sbpl_perception 
chmod +x data/scripts/dowload_demo_models.sh
./data/scripts/download_demo_models.sh
```

An example RGB-D scene containing 3 objects is provided under sbpl_perception/demo. To run PERCH on this with default parameters:

```bash
roscd sbpl_perception && mkdir visualization
roslaunch sbpl_perception demo.launch 
```

The states 'expanded' as well as the goal state will be saved under sbpl_perception/visualization. The expanded states will also show up on an OpenCV window named "Expansions". To also save all the 'generated' (rendered) states to sbpl_perception/visualization, use

```bash
roslaunch sbpl_perception demo.launch image_debug:=true
```
You should see the following input depth and output depth images under sbpl_perception/visualization:

![](https://cloud.githubusercontent.com/assets/1756204/15489006/ca12e31c-2129-11e6-9eed-4b984dd081fc.png)
![](https://cloud.githubusercontent.com/assets/1756204/15489005/ca10b7e0-2129-11e6-966e-b75c6a43ff3d.png)

Configuration parameters for the algorithm can be found under sbpl_perception/config/demo_env_config.yaml and sbpl_perception/config/demo_planner_config.yaml, along with descriptions of those parameters.

To pull changes from all related repositories in one go, run ```wstool update -t src```.
