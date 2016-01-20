#!/bin/bash
PERCH_ROOT=`rospack find sbpl_perception`
HEURISTICS_FOLDER="$PERCH_ROOT/heuristics"
DATA_FOLDER="$PERCH_ROOT/data"
SCENARIOS_FOLDER="$PERCH_ROOT/data/experiment_input"

roi_saver_exec="$PERCH_ROOT/bin/experiments/roi_saver" 

# Load rosparams and config variable to the parameter server
roslaunch "$PERCH_ROOT/config/household_models.xml"
rosparam load "$PERCH_ROOT/config/env_config.yaml"
rosparam load "$PERCH_ROOT/config/param_config.yaml"

for input_file in $SCENARIOS_FOLDER/*.txt; do
    echo $input_file
    $roi_saver_exec $input_file $HEURISTICS_FOLDER
done
