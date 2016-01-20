#!/bin/bash
PERCH_ROOT=`rospack find sbpl_perception`
HEURISTICS_FOLDER="$PERCH_ROOT/heuristics"
DATA_FOLDER="$PERCH_ROOT/data"
SCENARIOS_FOLDER="$PERCH_ROOT/data/experiment_input"

timestamp=`date "+%m_%d_%Y_%H_%M_%S"`

perch_exec="$PERCH_ROOT/bin/experiments/perch" 
poses_file="$PERCH_ROOT/experiments/perch_poses_$timestamp.txt"
stats_file="$PERCH_ROOT/experiments/perch_stats_$timestamp.txt"

# Load rosparams and config variable to the parameter server
roslaunch "$PERCH_ROOT/config/household_objects.xml"
rosparam load "$PERCH_ROOT/config/env_config.yaml" perch_experiments
rosparam load "$PERCH_ROOT/config/planner_config.yaml" perch_experiments

num_procs=8
i=0
for input_file in $SCENARIOS_FOLDER/*.txt; do
  let i=i+1
  # if [ "$i" == "1" ]; then
  #   continue
  # fi
  echo "mpirun -n $num_procs $perch_exec $input_file $poses_file $stats_file image_debug:=true"
  mpirun -n $num_procs $perch_exec $input_file $poses_file $stats_file image_debug:=false
done
