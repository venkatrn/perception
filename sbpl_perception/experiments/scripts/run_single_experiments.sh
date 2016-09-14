#!/bin/bash

timestamp=`date "+%m_%d_%Y_%H_%M_%S"`

PERCH_ROOT=`rospack find sbpl_perception`
HEURISTICS_FOLDER="$PERCH_ROOT/heuristics"
DATA_FOLDER="$PERCH_ROOT/data"
SCENARIOS_FOLDER="$PERCH_ROOT/data/experiment_input"
EXPERIMENTS_FOLDER="$PERCH_ROOT/experiments/results_$timestamp"
PERCH_EXEC=`catkin_find sbpl_perception perch_single_object`

function run_experiment() {
local poses_file=$1
local stats_file=$2
# i=0
for input_file in $SCENARIOS_FOLDER/*.txt; do
  echo "mpirun -n $NUM_PROCS $PERCH_EXEC $input_file $poses_file $stats_file image_debug:=false"
  # let i=i+1
  # if [ "$i" != "1" ]; then
  #   continue
  # fi
  mpirun -n $NUM_PROCS $PERCH_EXEC $input_file $poses_file $stats_file image_debug:=false
done
}

# Start roscore and wait to initialize
roscore &
sleep 2

# Load default rosparams and config variables to the parameter server. These can be overwritten
# later by the experiment-specific options.
roslaunch "$PERCH_ROOT/config/household_objects.xml"
rosparam load "$PERCH_ROOT/config/experiments_env_config.yaml" perch_experiments
rosparam load "$PERCH_ROOT/config/experiments_planner_config.yaml" perch_experiments

# If rcnn_heuristic_option is true, then we will run D2P using precomputed RCNN heuristics in the HEURISTICS_FOLDER,
# otherwise, we will run PERCH without RCNN heuristics.
# use_rcnn_heuristic_options="true false"
use_rcnn_heuristic_options="false"
rosparam set perch_experiments/perch_params/use_rcnn_heuristic false

# Run experiments for different epsilons
# epsilon_options="5 10"
rosparam set perch_experiments/inflation_epsilon 5

# Whether or not to use lazy edge evaluation.
# use_lazy_options="true false"
rosparam set perch_experiments/use_lazy false

# Different values for max_icp_iterations.
# max_icp_iterations_options="10 20 40"     
rosparam set perch_experiments/perch_params/max_icp_iterations 20

# Different options for translation and yaw resolutions.
# search_resolution_translation_options="0.02 0.04 0.1 0.2"
# search_resolution_yaw_options="0.3926991 0.27925268"
# search_resolution_translation_options="0.1"
# search_resolution_yaw_options="0.3926991"
# rosparam set perch_experiments/search_resolution_translation 0.1
rosparam set perch_experiments/search_resolution_translation 0.04
rosparam set perch_experiments/search_resolution_yaw 0.3926991

# Options for clutter_mode
# clutter_mode_options="true"
clutter_mode_options="false"
clutter_regularizer_options="0 0.2 0.4 0.6 0.8 1.0"

# Different options for number of processors to be used for parallelization.
# proc_options="5 10 20"
NUM_PROCS=8

if [[ ! -e $EXPERIMENTS_FOLDER ]]; then
  mkdir $EXPERIMENTS_FOLDER
fi

# Iterate over every combination of options.
# NOTE: Customize these per need!
for clutter_mode in $clutter_mode_options; do
  for clutter_regularizer in $clutter_regularizer_options; do

    rosparam set perch_experiments/perch_params/use_clutter_mode $clutter_mode
    rosparam set perch_experiments/perch_params/clutter_regularizer $clutter_regularizer


    # suffix=epsilon_$eps'_'icp_$icp'_'rcnn_$rcnn'_'lazy_$lazy'_'trans_$trans'_'yaw_$yaw'_'procs_$NUM_PROCS
    suffix=clutter_$clutter_mode'_'regularizer_$clutter_regularizer
    poses_file="$EXPERIMENTS_FOLDER/perch_poses_$suffix.txt"
    stats_file="$EXPERIMENTS_FOLDER/perch_stats_$suffix.txt"
    echo $poses_file

    # Run experiments for this configuration
    run_experiment $poses_file $stats_file

  done
done

# Terminate roscore
pkill roscore

