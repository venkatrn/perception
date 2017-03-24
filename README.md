# Deliberative Perception for Multi-Object Recognition and Localization

Overview
--------
This library provides implementations for single and multi-object instance localization from RGB-D sensor (MS Kinect, ASUS Xtion etc.) data. These are based on the <a href="http://www.cs.cmu.edu/~venkatrn/papers/icra16a.pdf">PERCH (Perception via Search)</a> and <a href="http://www.cs.cmu.edu/~venkatrn/papers/rss16.pdf">D2P (Discriminatively-guided Deliberative Perception)</a> algorithms.

Requirements
------------
- Ubuntu 14.04+
- ROS Hydro+ (active development only on Indigo)

Setup
-----

1. Get ROS Indigo from http://wiki.ros.org/indigo/Installation/Ubuntu
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
First, download the object CAD models (92 MB total):

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

For more information on setting up PERCH for your custom robot/sensor, reproducing experimental results, and API details, refer to the <a href="https://github.com/venkatrn/perception/wiki">wiki</a>.
