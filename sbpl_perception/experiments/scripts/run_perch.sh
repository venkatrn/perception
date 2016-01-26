#!/bin/bash

PERCH_ROOT=`rospack find sbpl_perception`
HEURISTICS_FOLDER="$PERCH_ROOT/heuristics"
DATA_FOLDER="$PERCH_ROOT/data"
SCENARIOS_FOLDER="$PERCH_ROOT/data/experiment_input"
PERCH_EXEC="$PERCH_ROOT/bin/experiments/perch" 

# NUM_PROCS=40

function run_experiment() {
  local poses_file=$1
  local stats_file=$2
	# i=0
  for input_file in $SCENARIOS_FOLDER/*.txt; do
    echo "mpirun -n $NUM_PROCS $PERCH_EXEC $input_file $poses_file $stats_file image_debug:=false"
    # let i=i+1
		# if [ "$i" != "1" ]; then
		# 	continue
		# fi
    mpirun -n $NUM_PROCS $PERCH_EXEC $input_file $poses_file $stats_file image_debug:=false
  done
}

timestamp=`date "+%m_%d_%Y_%H_%M_%S"`

# Load default rosparams and config variables to the parameter server
roslaunch "$PERCH_ROOT/config/household_objects.xml"
rosparam load "$PERCH_ROOT/config/env_config.yaml" perch_experiments
rosparam load "$PERCH_ROOT/config/planner_config.yaml" perch_experiments

epsilon_options="5 10"
use_rcnn_heuristic_options="true false"
use_lazy_options="false"
max_icp_iterations_options="10 20 40"                    
# search_resolution_translation_options="0.02 0.04 0.1 0.2"
search_resolution_translation_options="0.15 0.2"
search_resolution_yaw_options="0.3926991 0.27925268"
proc_options="5 10 20"

# for use_rcnn_heuristic in $use_rcnn_heuristic_options; do
# for epsilon in $epsilon_options; do
  # for search_resolution_translation in $search_resolution_translation_options; do
for use_lazy in $use_lazy_options; do
  for NUM_PROCS in $proc_options; do

    # rosparam set perch_experiments/inflation_epsilon $epsilon
    # rosparam set perch_experiments/perch_params/use_rcnn_heuristic $use_rcnn_heuristic
    # rosparam set perch_experiments/perch_params/max_icp_iterations $max_icp_iterations
    rosparam set perch_experiments/use_lazy $use_lazy
    # rosparam set perch_experiments/search_resolution_translation $search_resolution_translation
    # rosparam set perch_experiments/search_resolution_yaw $search_resolution_yaw

   icp=`rosparam get perch_experiments/perch_params/max_icp_iterations`
   rcnn=`rosparam get perch_experiments/perch_params/use_rcnn_heuristic`
   eps=`rosparam get perch_experiments/inflation_epsilon`
   lazy=`rosparam get perch_experiments/use_lazy`
   trans=`rosparam get perch_experiments/search_resolution_translation`
   yaw=`rosparam get perch_experiments/search_resolution_yaw`

   # suffix=epsilon_$eps'_'icp_$icp'_'rcnn_$rcnn'_'lazy_$lazy'_'trans_$trans'_'yaw_$yaw'_'
   suffix=epsilon_$eps'_'icp_$icp'_'rcnn_$rcnn'_'lazy_$lazy'_'trans_$trans'_'yaw_$yaw'_'procs_$NUM_PROCS
   poses_file="$PERCH_ROOT/experiments/perch_poses_$timestamp_$suffix.txt"
   stats_file="$PERCH_ROOT/experiments/perch_stats_$timestamp_$suffix.txt"

   # Run experiments for this configuration
   run_experiment $poses_file $stats_file

  done
done
