# Deliberative Perception for Multi-Object Recognition and Localization

Overview
--------
This repository provides the implementations for <a href="http://www.cs.cmu.edu/~venkatrn/papers/icra16a.pdf">PERCH (Perception via Search)</a> and <a href="http://www.cs.cmu.edu/~venkatrn/papers/rss16.pdf">D2P (Discriminatively-guided Deliberative Perception)</a>, algorithms for multi-object recognition and localization in RGB-D images. In addition, it contains a number of useful tools for dealing with RGB-D images.

The package 'kinect_sim' is based on PCL's simulation API and serves as the depth-image renderer/simulator for PERCH. This provides the functionality to generate the depth-image for a scene, given a camera pose, set of 3D models/meshes and their poses.

The 'sbpl_perception' package implements the PERCH and D2P algorithms. Internally, this works in conjunction with the SBPL planning library (https://github.com/sbpl/sbpl) which provides several domain-independent graph-search algorithms.

Dependencies
------------

- ROS Hydro+ (active development only on Indigo)
- C++11
- OpenCV 2.x
- PCL 1.7+

Setup
-----

1. Get ROS Indigo from http://wiki.ros.org/hydro/Installation/Ubuntu
2. Set up a catkin workspace ~/my_workspace (http://wiki.ros.org/catkin/Tutorials/create_a_workspace).
3. Download the <a href="https://raw.githubusercontent.com/venkatrn/perception/master/perch.rosinstall" download="perch.rosinstall">rosinstall file</a> to your workspace.

```bash
cd ~/my_workspace
wstool init src
wstool merge -t src perch.rosinstall
wstool update -t src
rosdep install --from-paths src --ignore-src --rosdistro indigo -y
catkin_make -DCMAKE_BUILD_TYPE=Release
```

Demo
----
First, download the RGB-D dataset and object CAD models (92 MB total):

```bash
roscd sbpl_perception 
chmod +x data/scripts/download_demo_models.sh
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

Running Experiments
-------------------

To reproduce results from our <a href="http://www.cs.cmu.edu/~venkatrn/papers/rss16.pdf">RSS '16 paper</a>, first checkout the RSS '16 version of the codebase:

```bash
git checkout tags/v1.0 
cd ~/my_workspace
catkin_make -DCMAKE_BUILD_TYPE=Release
```
Then run the experiments script:
```bash
roscd sbpl_perception 
chmod +x data/scripts/download_experiment_data.sh
./data/scripts/download_experiment_data.sh
./experiments/scripts/run_experiments.sh
```

This will a) download the test scenes to sbpl_perception/data/experiment_input and the precomputed RCNN heuristics to sbpl_perception/heuristics, b) run the experiments script to generate results for a run of lazy D2P with 8 processors for parallelization and other default parameters in the run_experiments.sh script. Statistical output will be saved to sbpl_perception/experiments/results_\<timestamp\>/ and visualization output (output depth images) will be saved to sbpl_perception/visualization/perch_poses_\<config_parameters\>/. You can modify the settings in run_experiments.sh to sweep across several combinations of parameters including lazy/non-lazy, use RCNN-heuristics/no heuristics, planner suboptimality bound, search resolution, maximum ICP iterations, and number of processors available for parallelization.


