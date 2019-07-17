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
Running with Python :
1. Clone ```https://github.com/SBPL-Cruz/maskrcnn-benchmark```
2. Create a python 3 virtual environment :
```
conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark
```
3. Install dependencies in requirements file
